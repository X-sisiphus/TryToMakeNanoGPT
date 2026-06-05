from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--sft-checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="out/sft_compare_samples")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop-at-eos", action="store_true")
    parser.add_argument("--stop-at-text", type=str, default=None)
    return parser.parse_args()

PROMPTS = [
    (
        "Explain GNSS",
        "Instruction:\n"
        "Explain the geodesy or astronomy concept in one paragraph.\n\n"
        "Input:\n"
        "GNSS time series\n\n"
        "Answer:\n"
    ),
    (
        "Extract Fields",
        "Instruction:\n"
        "Extract the station, signal, and unit from the text.\n\n"
        "Input:\n"
        "Station BJFS shows a vertical velocity of 2.4 mm/yr from GNSS observations.\n\n"
        "Answer:\n"
    ),
    (
        "Convert Format",
        "Instruction:\n"
        "Convert the observation into a compact JSON object.\n\n"
        "Input:\n"
        "VLBI baseline residual: station=WETTZELL, delay=12 ps, band=X.\n\n"
        "Answer:\n"
    ),
]

def run_sample(checkpoint, prompt, args):
    cmd = [
        sys.executable,
        str(ROOT / "sample.py"),
        "--checkpoint", checkpoint,
        "--prompt", prompt,
        "--max-new-tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--top-k", str(args.top_k),
        "--repetition-penalty", str(args.repetition_penalty),
    ]

    if args.stop_at_eos:
        cmd.append("--stop-at-eos")

    if args.stop_at_text is not None:
        cmd.extend(["--stop-at-text", args.stop_at_text])

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    output = result.stdout
    promptStart = output.rfind(prompt)

    if promptStart == -1:
        return output.strip()

    return output[promptStart + len(prompt):].strip()

def write_report(args, rows):
    os.makedirs(args.out_dir, exist_ok=True)
    outPath = os.path.join(args.out_dir, "report.md")

    with open(outPath, "w", encoding="utf-8") as f:
        f.write("# SFT Sampling Comparison\n\n")
        f.write(f"Base checkpoint: `{args.base_checkpoint}`\n\n")
        f.write(f"SFT checkpoint: `{args.sft_checkpoint}`\n\n")

        for title, prompt, baseText, sftText in rows:
            f.write(f"## {title}\n\n")
            f.write("### Prompt\n\n")
            f.write("```text\n")
            f.write(prompt)
            f.write("\n```\n\n")

            f.write("### Continued Pretraining\n\n")
            f.write("```text\n")
            f.write(baseText)
            f.write("\n```\n\n")

            f.write("### SFT\n\n")
            f.write("```text\n")
            f.write(sftText)
            f.write("\n```\n\n")

    print(f"saved report to {outPath}")

def main():
    args = parse_args()
    rows = []

    for title, prompt in PROMPTS:
        print(f"sampling: {title} / base", flush=True)
        baseText = run_sample(args.base_checkpoint, prompt, args)

        print(f"sampling: {title} / sft", flush=True)
        sftText = run_sample(args.sft_checkpoint, prompt, args)

        rows.append((title, prompt, baseText, sftText))

    write_report(args, rows)


if __name__ == "__main__":
    main()
