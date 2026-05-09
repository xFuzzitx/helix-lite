"""Multi-needle NIAH harness — RULER-style retrieval eval.

For each context length, place N distinct needle-question pairs at
random depths in a long filler text, ask all N questions in one shot
at the end, and grade the answers.

Each needle is a unique key:value pair like:
  "The tropical fruit code for Bahrain is XQ-7392."
And the question is:
  "What is the tropical fruit code for Bahrain?"
The answer is graded as correct if the needle's value substring
(``XQ-7392``) appears in the model's reply for that question.

Score format:
  per-context-length recall = (found needles) / (total needles).

Why this matters: single-needle NIAH is trivial at 128K for a
1M-context model. Multi-needle is what shows up as the real signal
on RULER and MRCR-V2 - it stresses simultaneous retrieval through a
long context.

Usage:
    PYTHONPATH=src python benchmarks/quality/run_multi_needle.py \\
        --model graelo/Qwen2.5-7B-Instruct-1M-AWQ \\
        --tp 1 --gmu 0.85 \\
        --ctx 32000 128000 \\
        --num-needles 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# A small fixed pool of needle templates; each pick instantiates
# one (key, value, question) triple.
NEEDLE_POOL = [
    ("The tropical fruit code for Bahrain is {value}.",
     "What is the tropical fruit code for Bahrain?"),
    ("The desert wind speed in Tatooine measures {value} kph.",
     "What is the desert wind speed in Tatooine in kph?"),
    ("The migration corridor for swallows is highway {value}.",
     "Which highway is the migration corridor for swallows?"),
    ("The encryption salt for project Orion is {value}.",
     "What is the encryption salt for project Orion?"),
    ("The harvest yield in plot 47 reached {value} kilograms.",
     "What was the harvest yield in plot 47, in kilograms?"),
    ("The submarine call sign of the Nautilus is {value}.",
     "What is the submarine call sign of the Nautilus?"),
    ("The library shelf for forbidden manuscripts is {value}.",
     "Which shelf holds the forbidden manuscripts?"),
    ("The patron color of the spring festival is {value}.",
     "What is the patron color of the spring festival?"),
    ("The moon phase ratio at solstice equals {value}.",
     "What is the moon phase ratio at solstice?"),
    ("The genetic marker for hybrid roses is {value}.",
     "What is the genetic marker for hybrid roses?"),
    ("The signal frequency for the Vault is {value} MHz.",
     "What is the signal frequency for the Vault in MHz?"),
    ("The recipe ratio for Marquis butter is {value}.",
     "What is the recipe ratio for Marquis butter?"),
]


def make_value(rng: random.Random) -> str:
    """A short, distinctive token unlikely to appear in filler."""
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return f"{rng.choice(letters)}{rng.choice(letters)}-{rng.randrange(1000, 9999)}"


def build_prompt(target_tokens: int, num_needles: int, tokenizer,
                 seed: int = 0) -> tuple[str, list[dict]]:
    """Return (prompt, needles list) where each needle is a dict with
    keys: ``template``, ``question``, ``value``, ``approx_depth``."""
    rng = random.Random(seed)
    pool = list(NEEDLE_POOL)
    rng.shuffle(pool)
    pool = pool[:num_needles]

    needles = []
    for tpl, q in pool:
        val = make_value(rng)
        needles.append(
            {"template": tpl, "question": q, "value": val,
             "filled": tpl.format(value=val)}
        )

    # Build filler then weave needles in at evenly-spaced depths.
    filler_unit = "The river follows the valley to the sea, slow and silver in the dawn light. " * 60 + "\n\n"
    chars = target_tokens * 4 - sum(len(n["filled"]) + 4 for n in needles) - 800
    filler_text = (filler_unit * (chars // len(filler_unit) + 1))[:max(chars, 0)]

    # Insert needles at evenly-spaced positions, with jitter
    n = len(needles)
    body_len = len(filler_text)
    insert_positions = []
    for i in range(n):
        depth = (i + 1) / (n + 1)
        jitter = rng.uniform(-0.05, 0.05)
        insert_positions.append(int(body_len * max(0.02, min(0.98, depth + jitter))))
    insert_positions.sort()

    parts = []
    last = 0
    for pos, needle in zip(insert_positions, needles):
        parts.append(filler_text[last:pos])
        parts.append(f"\n\n{needle['filled']}\n\n")
        last = pos
        needle["approx_depth"] = pos / max(1, body_len)
    parts.append(filler_text[last:])
    body = "".join(parts)

    # Append the question block at the end
    questions_block = (
        "\n\n---\n"
        "Below are several questions about specific facts mentioned in the "
        "passage above. Answer each one with the requested value only, on "
        "its own line, in the format 'Q<N>: <value>'.\n\n"
    )
    for i, n_ in enumerate(needles, start=1):
        questions_block += f"Q{i}: {n_['question']}\nA{i}: "
        if i < len(needles):
            questions_block += "\n"
    questions_block += "\n\nReply now:\n"

    prompt = body + questions_block
    return prompt, needles


def grade(completion: str, needles: list[dict]) -> list[bool]:
    """Per-needle pass/fail by substring match on the value."""
    return [n["value"] in completion for n in needles]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="graelo/Qwen2.5-7B-Instruct-1M-AWQ")
    p.add_argument("--ctx", nargs="+", type=int, default=[32000, 128000])
    p.add_argument("--max-ctx", type=int, default=128_000)
    p.add_argument("--num-needles", type=int, default=8)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gmu", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=400)
    args = p.parse_args()

    if args.tp > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.tp))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print(f"=== multi-needle NIAH ({args.num_needles} needles) ===")
    print(f"  model={args.model}  TP={args.tp}  gmu={args.gmu}")
    print(f"  contexts={args.ctx}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        dtype="auto",
        max_model_len=args.max_ctx,
        gpu_memory_utilization=args.gmu,
        enforce_eager=True,
        trust_remote_code=False,
        tensor_parallel_size=args.tp,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, top_p=1.0)

    results = []
    for ctx in args.ctx:
        if ctx > args.max_ctx:
            print(f"  skip {ctx}: exceeds max-ctx {args.max_ctx}")
            continue
        prompt, needles = build_prompt(ctx, args.num_needles, tok, seed=args.seed)
        actual = len(tok.encode(prompt))
        print(f"\n--- ctx={ctx:,} (actual {actual:,} tok) ---")
        t0 = time.time()
        out = llm.generate([prompt], sampling)
        elapsed = time.time() - t0
        completion = out[0].outputs[0].text
        verdicts = grade(completion, needles)
        recall = sum(verdicts) / len(verdicts)
        print(f"  elapsed={elapsed:.1f}s  recall={sum(verdicts)}/{len(verdicts)} ({100*recall:.0f}%)")
        for i, (n_, ok) in enumerate(zip(needles, verdicts), start=1):
            mark = "✓" if ok else "✗"
            print(f"  {mark} Q{i} value={n_['value']!r:>10} depth={n_['approx_depth']:.2f}")
        # Save the raw completion for inspection
        results.append({
            "ctx_target": ctx, "ctx_actual": actual, "elapsed_s": elapsed,
            "recall": recall, "found": sum(verdicts), "total": len(verdicts),
            "needles": [{"value": n_["value"], "depth": n_["approx_depth"],
                          "found": ok} for n_, ok in zip(needles, verdicts)],
            "completion_preview": completion[:600],
        })

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = args.model.replace("/", "_")
    out_path = RESULTS_DIR / f"multi_needle_{safe}_tp{args.tp}_{ts}.json"
    out_path.write_text(json.dumps({
        "stage": "multi-needle",
        "model": args.model, "tp": args.tp, "gmu": args.gmu,
        "num_needles": args.num_needles, "seed": args.seed,
        "ts": ts, "results": results,
    }, indent=2))
    print(f"\n✓ saved {out_path}")
    print("\nSummary:")
    print(f"  {'ctx':>10}  recall  elapsed")
    for r in results:
        print(f"  {r['ctx_target']:>10,}  {r['found']}/{r['total']:>2}  {r['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
