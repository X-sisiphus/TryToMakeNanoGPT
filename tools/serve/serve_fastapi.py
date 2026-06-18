from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import asyncio
import os
import statistics
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import tiktoken
import uvicorn

from model import BigramLanguageModel, GPTConfig


class GenerateRequest(BaseModel):
    prompt: str = Field(default="\n")
    max_new_tokens: int = Field(default=80, ge=1, le=512)
    temperature: float = Field(default=1.0, gt=0.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    repetition_penalty: float = Field(default=1.0, ge=1.0)
    stop_at_eos: bool = Field(default=False)
    stop_at_text: Optional[str] = Field(default=None)
    use_kv_cache: bool = Field(default=False)


class GenerateBatchRequest(BaseModel):
    prompts: List[str] = Field(min_length=1, max_length=32)
    max_new_tokens: int = Field(default=80, ge=1, le=512)
    temperature: float = Field(default=1.0, gt=0.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    repetition_penalty: float = Field(default=1.0, ge=1.0)
    stop_at_eos: bool = Field(default=False)
    stop_at_text: Optional[str] = Field(default=None)
    use_kv_cache: bool = Field(default=False)


class DynamicBatchItem:
    def __init__(self, request, future, enqueueTime):
        self.request = request
        self.future = future
        self.enqueueTime = enqueueTime


class ModelServer:
    def __init__(self, checkpointPath, device):
        self.checkpointPath = checkpointPath
        self.device = device
        self.model, self.encode, self.decode, self.eosTokenId, self.padTokenId, self.vocabType = self.load_model(
            checkpointPath,
            device,
        )

    def load_model(self, checkpointPath, device):
        checkpoint = torch.load(
            checkpointPath,
            map_location=device,
            weights_only=False,
        )
        config = GPTConfig(**checkpoint["config"])

        model = BigramLanguageModel(
            config.vocabSize,
            config.blockSize,
            config=config,
        )
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()

        vocabInfo = checkpoint["vocab"]
        vocabType = vocabInfo.get("type", "char")
        eosTokenId = None
        padTokenId = 0

        if vocabType == "tokenizer":
            enc = tiktoken.get_encoding(vocabInfo["meta"]["encoding"])
            eosTokenId = enc.eot_token
            padTokenId = eosTokenId

            def encode(text):
                return enc.encode(text)

            def decode(tokenIds):
                return enc.decode(tokenIds)

        elif vocabType == "char":
            stringToInt = vocabInfo["stringToInt"]
            intToString = vocabInfo["intToString"]
            padTokenId = 0

            def encode(text):
                try:
                    return [stringToInt[c] for c in text]
                except KeyError as exc:
                    raise ValueError(f"checkpoint 词表中不存在字符: {exc}") from exc

            def decode(tokenIds):
                return "".join([intToString[i] for i in tokenIds])

        else:
            raise ValueError(f"不支持的 vocab type: {vocabType}")

        return model, encode, decode, eosTokenId, padTokenId, vocabType

    def sync_if_needed(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def generate(self, request):
        try:
            promptIds = self.encode(request.prompt)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if len(promptIds) == 0:
            raise HTTPException(status_code=400, detail="prompt 编码后为空")

        context = torch.tensor(
            [promptIds],
            dtype=torch.long,
            device=self.device,
        )
        if (
            request.use_kv_cache
            and not self.model.config.useRoPE
            and context.shape[1] + request.max_new_tokens > self.model.config.blockSize
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "非 RoPE 模型 use_kv_cache=True 时，prompt_tokens + max_new_tokens "
                    f"不能超过 block_size={self.model.config.blockSize}"
                ),
            )

        eosTokenId = self.eosTokenId if request.stop_at_eos else None

        self.sync_if_needed()
        start = time.perf_counter()
        generated = self.model.generate(
            context,
            request.max_new_tokens,
            temperature=request.temperature,
            topK=request.top_k,
            repetitionPenalty=request.repetition_penalty,
            repetitionStart=context.shape[1],
            eosTokenId=eosTokenId,
            useKvCache=request.use_kv_cache,
        )
        self.sync_if_needed()
        latency = time.perf_counter() - start

        generatedIds = generated[0].tolist()
        if request.stop_at_eos and eosTokenId is not None:
            generatedTail = generatedIds[context.shape[1] :]
            if eosTokenId in generatedTail:
                eosPos = generatedIds.index(eosTokenId, context.shape[1])
                generatedIds = generatedIds[:eosPos]

        text = self.decode(generatedIds)
        if request.stop_at_text is not None and request.stop_at_text in text:
            text = text.split(request.stop_at_text)[0]

        completionText = text[len(request.prompt) :] if text.startswith(request.prompt) else text
        totalTokens = len(generatedIds)
        newTokens = max(0, totalTokens - len(promptIds))

        return {
            "text": text,
            "completion_text": completionText,
            "prompt_tokens": len(promptIds),
            "new_tokens": newTokens,
            "total_tokens": totalTokens,
            "latency_sec": latency,
            "tokens_per_sec": newTokens / latency if latency > 0 else 0.0,
            "device": self.device,
            "vocab_type": self.vocabType,
            "checkpoint": self.checkpointPath,
            "use_kv_cache": request.use_kv_cache,
        }

    @torch.no_grad()
    def generate_batch(self, request):
        try:
            promptIdsList = [self.encode(prompt) for prompt in request.prompts]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if any(len(promptIds) == 0 for promptIds in promptIdsList):
            raise HTTPException(status_code=400, detail="prompt 编码后为空")

        promptLengths = [len(promptIds) for promptIds in promptIdsList]

        if (
            request.use_kv_cache
            and not self.model.config.useRoPE
            and promptLengths[0] + request.max_new_tokens > self.model.config.blockSize
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "非 RoPE 模型 use_kv_cache=True 时，prompt_tokens + max_new_tokens "
                    f"不能超过 block_size={self.model.config.blockSize}"
                ),
            )

        maxPromptLen = max(promptLengths)
        paddedPromptIdsList = []
        attentionMaskList = []

        for promptIds in promptIdsList:
            padLen = maxPromptLen - len(promptIds)
            paddedPromptIdsList.append([self.padTokenId] * padLen + promptIds)
            attentionMaskList.append([0] * padLen + [1] * len(promptIds))

        context = torch.tensor(
            paddedPromptIdsList,
            dtype=torch.long,
            device=self.device,
        )
        attentionMask = torch.tensor(
            attentionMaskList,
            dtype=torch.long,
            device=self.device,
        )
        eosTokenId = self.eosTokenId if request.stop_at_eos else None

        self.sync_if_needed()
        start = time.perf_counter()
        generated = self.model.generate(
            context,
            request.max_new_tokens,
            temperature=request.temperature,
            topK=request.top_k,
            repetitionPenalty=request.repetition_penalty,
            repetitionStart=context.shape[1],
            eosTokenId=eosTokenId,
            useKvCache=request.use_kv_cache,
            attentionMask=attentionMask,
        )
        self.sync_if_needed()
        latency = time.perf_counter() - start

        outputs = []
        totalNewTokens = 0

        for rowIdx, prompt in enumerate(request.prompts):
            generatedIds = generated[rowIdx].tolist()
            promptLen = promptLengths[rowIdx]
            padLen = maxPromptLen - promptLen
            generatedIds = generatedIds[padLen:]

            if request.stop_at_eos and eosTokenId is not None:
                generatedTail = generatedIds[promptLen:]
                if eosTokenId in generatedTail:
                    eosPos = generatedIds.index(eosTokenId, promptLen)
                    generatedIds = generatedIds[:eosPos]

            text = self.decode(generatedIds)
            if request.stop_at_text is not None and request.stop_at_text in text:
                text = text.split(request.stop_at_text)[0]

            completionText = text[len(prompt) :] if text.startswith(prompt) else text
            newTokens = max(0, len(generatedIds) - promptLen)
            totalNewTokens += newTokens

            outputs.append(
                {
                    "text": text,
                    "completion_text": completionText,
                    "prompt_tokens": promptLen,
                    "new_tokens": newTokens,
                    "total_tokens": len(generatedIds),
                }
            )

        return {
            "outputs": outputs,
            "batch_size": len(request.prompts),
            "latency_sec": latency,
            "total_new_tokens": totalNewTokens,
            "tokens_per_sec": totalNewTokens / latency if latency > 0 else 0.0,
            "device": self.device,
            "vocab_type": self.vocabType,
            "checkpoint": self.checkpointPath,
            "use_kv_cache": request.use_kv_cache,
        }


class DynamicBatcher:
    def __init__(self, server, maxBatchSize=8, waitMs=5):
        self.server = server
        self.maxBatchSize = maxBatchSize
        self.waitSec = waitMs / 1000.0
        self.pendingItems = []
        self.flushTask = None
        self.lock = asyncio.Lock()
        self.statsLock = asyncio.Lock()
        self.reset_stats()

    def reset_stats(self):
        self.totalRequests = 0
        self.totalBatches = 0
        self.batchSizes = []
        self.queueWaitMs = []
        self.batchLatencyMs = []

    def is_compatible(self, left, right):
        return (
            left.max_new_tokens == right.max_new_tokens
            and left.temperature == right.temperature
            and left.top_k == right.top_k
            and left.repetition_penalty == right.repetition_penalty
            and left.stop_at_eos == right.stop_at_eos
            and left.stop_at_text == right.stop_at_text
            and left.use_kv_cache == right.use_kv_cache
        )

    async def generate(self, request):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = DynamicBatchItem(request, future, time.perf_counter())

        async with self.lock:
            self.pendingItems.append(item)
            if self.flushTask is None or self.flushTask.done():
                self.flushTask = asyncio.create_task(self.flush_after_wait())

        return await future

    async def flush_after_wait(self):
        await asyncio.sleep(self.waitSec)

        async with self.lock:
            items = self.pendingItems
            self.pendingItems = []

        while items:
            firstItem = items.pop(0)

            batchItems = [firstItem]
            remainingItems = []
            for item in items:
                if self.is_compatible(firstItem.request, item.request):
                    if len(batchItems) < self.maxBatchSize:
                        batchItems.append(item)
                    else:
                        remainingItems.append(item)
                else:
                    remainingItems.append(item)

            await self.run_batch(batchItems)
            items = remainingItems

    async def run_batch(self, batchItems):
        firstRequest = batchItems[0].request
        batchRequest = GenerateBatchRequest(
            prompts=[item.request.prompt for item in batchItems],
            max_new_tokens=firstRequest.max_new_tokens,
            temperature=firstRequest.temperature,
            top_k=firstRequest.top_k,
            repetition_penalty=firstRequest.repetition_penalty,
            stop_at_eos=firstRequest.stop_at_eos,
            stop_at_text=firstRequest.stop_at_text,
            use_kv_cache=firstRequest.use_kv_cache,
        )

        try:
            batchStartTime = time.perf_counter()
            batchResponse = await asyncio.to_thread(
                self.server.generate_batch,
                batchRequest,
            )
            finishTime = time.perf_counter()
            batchLatencyMs = (finishTime - batchStartTime) * 1000
            batchSize = batchResponse["batch_size"]
            waitMsList = [
                max(0.0, (batchStartTime - item.enqueueTime) * 1000)
                for item in batchItems
            ]

            async with self.statsLock:
                self.totalRequests += len(batchItems)
                self.totalBatches += 1
                self.batchSizes.append(batchSize)
                self.queueWaitMs.extend(waitMsList)
                self.batchLatencyMs.append(batchLatencyMs)

            for item, output in zip(batchItems, batchResponse["outputs"]):
                latency = finishTime - item.enqueueTime
                queueWaitMs = max(0.0, (batchStartTime - item.enqueueTime) * 1000)
                result = {
                    **output,
                    "latency_sec": latency,
                    "tokens_per_sec": output["new_tokens"] / latency if latency > 0 else 0.0,
                    "device": batchResponse["device"],
                    "vocab_type": batchResponse["vocab_type"],
                    "checkpoint": batchResponse["checkpoint"],
                    "use_kv_cache": batchResponse["use_kv_cache"],
                    "dynamic_batch_size": batchSize,
                    "dynamic_wait_ms": self.waitSec * 1000,
                    "queue_wait_ms": queueWaitMs,
                    "batch_latency_ms": batchLatencyMs,
                }
                item.future.set_result(result)
        except Exception as exc:
            for item in batchItems:
                item.future.set_exception(exc)

    async def get_stats(self):
        async with self.statsLock:
            batchSizes = list(self.batchSizes)
            queueWaitMs = list(self.queueWaitMs)
            batchLatencyMs = list(self.batchLatencyMs)
            totalRequests = self.totalRequests
            totalBatches = self.totalBatches

        if totalBatches == 0:
            return {
                "total_requests": 0,
                "total_batches": 0,
                "max_batch_size": self.maxBatchSize,
                "wait_ms": self.waitSec * 1000,
            }

        return {
            "total_requests": totalRequests,
            "total_batches": totalBatches,
            "max_batch_size": self.maxBatchSize,
            "wait_ms": self.waitSec * 1000,
            "avg_batch_size": statistics.mean(batchSizes),
            "max_observed_batch_size": max(batchSizes),
            "avg_queue_wait_ms": statistics.mean(queueWaitMs) if queueWaitMs else 0.0,
            "avg_batch_latency_ms": statistics.mean(batchLatencyMs),
            "batch_size_histogram": {
                str(size): batchSizes.count(size)
                for size in sorted(set(batchSizes))
            },
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--use-mps", action="store_true")
    return parser.parse_args()


def create_app(server):
    app = FastAPI(title="nanoGPT Inference Server")
    dynamicBatcher = DynamicBatcher(server)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "device": server.device,
            "vocab_type": server.vocabType,
            "checkpoint": server.checkpointPath,
            "parameters": server.model.get_num_params(),
            "block_size": server.model.config.blockSize,
        }

    @app.post("/generate")
    def generate(request: GenerateRequest):
        return server.generate(request)

    @app.post("/generate_dynamic")
    async def generate_dynamic(request: GenerateRequest):
        return await dynamicBatcher.generate(request)

    @app.get("/dynamic_stats")
    async def dynamic_stats():
        return await dynamicBatcher.get_stats()

    @app.post("/generate_batch")
    def generate_batch(request: GenerateBatchRequest):
        return server.generate_batch(request)

    return app


def main():
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() and args.use_mps else "cpu"
    server = ModelServer(args.checkpoint, device)
    app = create_app(server)

    print(f"loaded checkpoint: {args.checkpoint}", flush=True)
    print(f"using device: {device}", flush=True)
    print(f"vocab type: {server.vocabType}", flush=True)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
