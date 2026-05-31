import argparse
import torch
from model import BigramLanguageModel, GPTConfig
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--start", type=str, default="\n")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
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
stringToInt = checkpoint["vocab"]["stringToInt"]
intToString = checkpoint["vocab"]["intToString"]

def encode(s):
    return [stringToInt[c] for c in s]

def decode(ids):
    return "".join([intToString[i] for i in ids])

context = torch.tensor(
    [encode(args.start)],
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

print(decode(generated[0].tolist()))