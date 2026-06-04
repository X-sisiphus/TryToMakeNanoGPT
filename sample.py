import argparse
import torch
from model import BigramLanguageModel, GPTConfig
import os
import tiktoken

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--start", type=str, default="\n")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stop-at-eos", action="store_true")
    return parser.parse_args()

args = parse_args()
useMps = os.environ.get("USE_MPS") == "1"
device = "mps" if torch.backends.mps.is_available() and useMps else "cpu"
print(f"using {device}", flush=True)

checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
config = GPTConfig(**checkpoint["config"])

model = BigramLanguageModel(config.vocabSize, config.blockSize, config=config)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()
print(f"number of parameters: {model.get_num_params() / 1e6:.2f}M", flush=True)
vocabInfo = checkpoint["vocab"]
vocabType = vocabInfo.get("type", "char")
startText = args.prompt if args.prompt is not None else args.start
print(f"vocab type: {vocabType}", flush=True)
eosId = None

if vocabType == "tokenizer":
    meta = vocabInfo["meta"]
    enc = tiktoken.get_encoding(meta["encoding"])
    eosId = enc.eot_token

    def encode(s):
        return enc.encode(s)

    def decode(ids):
        return enc.decode(ids)

elif vocabType == "char":
    stringToInt = vocabInfo["stringToInt"]
    intToString = vocabInfo["intToString"]

    def encode(s):
        return [stringToInt[c] for c in s]

    def decode(ids):
        return "".join([intToString[i] for i in ids])

else:
    raise ValueError(f"不支持的 vocab type: {vocabType}")

context = torch.tensor(
    [encode(startText)],
    dtype=torch.long,
    device=device,
)

with torch.no_grad():
    generated = model.generate(
        context,
        args.max_new_tokens,
        temperature=args.temperature,
        topK=args.top_k,
    )

generatedIds = generated[0].tolist()

if args.stop_at_eos and eosId is not None:
    promptLen = context.shape[1]
    generatedTail = generatedIds[promptLen:]

    if eosId in generatedTail:
        eosPos = generatedIds.index(eosId, promptLen)
        generatedIds = generatedIds[:eosPos]

print(decode(generatedIds))
