"""
HELIX-Lite — baseline benchmark.

Loads Qwen2.5-7B-Instruct-1M, measures memory + throughput at increasing context
lengths. This is the reference point against which all PRs are compared.

Outputs JSON to benchmarks/results/baseline_<timestamp>.json

Usage:
    source .venv/bin/activate
    python benchmarks/run_baseline.py [--max-ctx 128000] [--gpu 0]
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-1M"
RESULTS_DIR = Path(__file__).parent / "results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-ctx", type=int, default=128_000,
                   help="max context length to test (default: 128K)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index to use when --tp=1 (default: 0)")
    p.add_argument("--tp", type=int, default=1,
                   help="tensor parallel size; --tp=2 splits weights+KV across "
                        "GPUs 0 and 1 (default: 1)")
    p.add_argument("--gmu", type=float, default=0.85,
                   help="gpu_memory_utilization (default: 0.85)")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--quick", action="store_true",
                   help="quick run: only 4K and 32K, no 128K")
    return p.parse_args()


def measure_memory(gpu: int) -> dict:
    free, total = torch.cuda.mem_get_info(gpu)
    return {
        "gpu": gpu,
        "free_gb": free / 1e9,
        "total_gb": total / 1e9,
        "used_gb": (total - free) / 1e9,
    }


def make_long_prompt(tokens: int, tokenizer) -> str:
    """Generate a long prompt with a needle for sanity-check NIAH."""
    needle = "The secret password is BANANA-7392."
    filler = ("The quick brown fox jumps over the lazy dog. " * 100 + "\n")
    # rough char-to-token ratio for English ≈ 4 chars/token
    prompt_chars = tokens * 4
    body = (filler * (prompt_chars // len(filler) + 1))[:prompt_chars - 200]
    # insert needle at ~50% depth
    mid = len(body) // 2
    body = body[:mid] + "\n\n" + needle + "\n\n" + body[mid:]
    body += "\n\nQuestion: What is the secret password? Answer in one word: "
    return body


def benchmark_length(llm, tokenizer, ctx: int, gpu: int) -> dict:
    """Run a single context-length benchmark."""
    from vllm import SamplingParams

    print(f"\n=== Context: {ctx:,} tokens ===")
    mem_before = measure_memory(gpu)
    print(f"  GPU mem before: {mem_before['used_gb']:.2f} GB used / "
          f"{mem_before['total_gb']:.2f} GB total")

    prompt = make_long_prompt(ctx, tokenizer)
    actual_tokens = len(tokenizer.encode(prompt))
    print(f"  prompt tokens: {actual_tokens:,} (target {ctx:,})")

    sampling = SamplingParams(temperature=0.0, max_tokens=16, top_p=1.0)

    # prefill timing
    t0 = time.time()
    outputs = llm.generate([prompt], sampling)
    elapsed = time.time() - t0

    mem_after = measure_memory(gpu)
    print(f"  GPU mem after:  {mem_after['used_gb']:.2f} GB used")
    print(f"  prefill+decode elapsed: {elapsed:.2f}s")
    print(f"  throughput: {(actual_tokens + 16) / elapsed:,.0f} tok/s")

    completion = outputs[0].outputs[0].text.strip()
    needle_found = "BANANA-7392" in completion or "BANANA" in completion
    print(f"  needle found: {needle_found}")
    print(f"  completion: {completion[:100]!r}")

    return {
        "context_tokens_target": ctx,
        "context_tokens_actual": actual_tokens,
        "elapsed_s": elapsed,
        "throughput_tok_s": (actual_tokens + 16) / elapsed,
        "mem_used_gb_before": mem_before["used_gb"],
        "mem_used_gb_after": mem_after["used_gb"],
        "kv_cache_gb": mem_after["used_gb"] - mem_before["used_gb"],
        "needle_found": needle_found,
        "completion_preview": completion[:200],
    }


def main():
    args = parse_args()

    if args.tp > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.tp))
        gpu_for_mem = 0  # local index after CUDA_VISIBLE_DEVICES restriction
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        gpu_for_mem = 0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" HELIX-Lite — baseline benchmark")
    print("=" * 60)
    print(f" Model:  {MODEL_ID}")
    print(f" TP:     {args.tp} (GPUs: {os.environ['CUDA_VISIBLE_DEVICES']})")
    print(f" Dtype:  {args.dtype}")
    print(f" Max ctx: {args.max_ctx:,}")
    print(f" gmu:    {args.gmu}")
    print("=" * 60)

    # Lazy import — vLLM startup is slow; fail fast on missing deps first
    from transformers import AutoTokenizer
    from vllm import LLM

    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("\n[2] Loading model into vLLM (downloads weights on first run, ~14 GB)...")
    # max_model_len is the hard cap; vLLM allocates KV scratch up to this
    # enforce_eager=True: disables CUDA graphs, which segfault on sm_86 + FLASHINFER
    # in vLLM 0.20.1 with long max_seq. -10 to 20% throughput trade-off, but stable.
    # No attention_backend kwarg: let vLLM auto-pick (FLASH_ATTN works once DCA is stripped).
    llm = LLM(
        model=MODEL_ID,
        dtype=args.dtype,
        max_model_len=args.max_ctx,
        gpu_memory_utilization=args.gmu,
        enforce_eager=True,
        trust_remote_code=False,
        tensor_parallel_size=args.tp,
    )
    print("  ✓ model loaded")

    # Pick context lengths to test
    if args.quick:
        contexts = [4_000, 32_000]
    else:
        contexts = [4_000, 32_000, 128_000]
        if args.max_ctx >= 256_000:
            contexts.append(256_000)
        if args.max_ctx >= 1_000_000:
            contexts.append(1_000_000)

    contexts = [c for c in contexts if c <= args.max_ctx]

    print(f"\n[3] Running benchmarks at: {contexts}")
    results = []
    for ctx in contexts:
        try:
            r = benchmark_length(llm, tokenizer, ctx, gpu_for_mem)
            results.append(r)
        except Exception as e:
            print(f"  ✗ FAILED at {ctx:,}: {e}")
            results.append({"context_tokens_target": ctx, "error": str(e)})

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"baseline_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL_ID,
            "tp": args.tp,
            "max_ctx": args.max_ctx,
            "gmu": args.gmu,
            "dtype": args.dtype,
            "timestamp": timestamp,
            "results": results,
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(f" ✓ Done. Results saved to {out_path}")
    print("=" * 60)
    print("\nSummary:")
    print(f"  {'context':>12}  {'mem (GB)':>10}  {'KV (GB)':>9}  {'throughput':>14}  needle")
    for r in results:
        if "error" in r:
            print(f"  {r['context_tokens_target']:>12,}  {'FAILED':>10}  {'-':>9}  {'-':>14}  -")
        else:
            print(f"  {r['context_tokens_target']:>12,}  "
                  f"{r['mem_used_gb_after']:>10.2f}  "
                  f"{r['kv_cache_gb']:>9.2f}  "
                  f"{r['throughput_tok_s']:>10,.0f} t/s  "
                  f"{'✓' if r['needle_found'] else '✗'}")


if __name__ == "__main__":
    main()
