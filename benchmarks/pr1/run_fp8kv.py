"""PR1a — fp8 KV cache baseline (built-in vLLM kv_cache_dtype='fp8'). 2x compression."""
import argparse, json, os, time
from datetime import datetime
from pathlib import Path
import torch

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-1M"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def make_long_prompt(tokens, tokenizer):
    needle = "The secret password is BANANA-7392."
    filler = ("The quick brown fox jumps over the lazy dog. " * 100 + "\n")
    prompt_chars = tokens * 4
    body = (filler * (prompt_chars // len(filler) + 1))[:prompt_chars - 200]
    mid = len(body) // 2
    body = body[:mid] + "\n\n" + needle + "\n\n" + body[mid:]
    body += "\n\nQuestion: What is the secret password? Answer in one word: "
    return body


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-ctx", type=int, default=128_000)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gmu", type=float, default=0.85)
    p.add_argument("--kv-dtype", default="fp8", choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"])
    args = p.parse_args()

    if args.tp > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.tp))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== PR1a fp8 KV baseline ===")
    print(f"  TP={args.tp} max_ctx={args.max_ctx} gmu={args.gmu} kv_dtype={args.kv_dtype}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        max_model_len=args.max_ctx,
        gpu_memory_utilization=args.gmu,
        enforce_eager=True,
        trust_remote_code=False,
        tensor_parallel_size=args.tp,
        kv_cache_dtype=args.kv_dtype,
    )

    contexts = [4_000, 32_000, 128_000]
    contexts = [c for c in contexts if c <= args.max_ctx]

    results = []
    sampling = SamplingParams(temperature=0.0, max_tokens=16, top_p=1.0)
    for ctx in contexts:
        try:
            free, total = torch.cuda.mem_get_info(0)
            mem_before = (total - free) / 1e9
            prompt = make_long_prompt(ctx, tokenizer)
            actual = len(tokenizer.encode(prompt))
            print(f"\n--- ctx={ctx:,} (actual {actual:,} tok) ---")
            t0 = time.time()
            out = llm.generate([prompt], sampling)
            elapsed = time.time() - t0
            free, total = torch.cuda.mem_get_info(0)
            mem_after = (total - free) / 1e9
            comp = out[0].outputs[0].text.strip()
            found = "BANANA-7392" in comp or "BANANA" in comp.upper()
            r = {
                "ctx_target": ctx, "ctx_actual": actual,
                "elapsed_s": elapsed, "throughput_tok_s": (actual + 16) / elapsed,
                "mem_before_gb": mem_before, "mem_after_gb": mem_after,
                "kv_delta_gb": mem_after - mem_before,
                "needle_found": found, "completion": comp[:120],
            }
            print(f"  elapsed={elapsed:.1f}s  throughput={(actual+16)/elapsed:.0f} t/s")
            print(f"  mem={mem_after:.2f} GB (Δ {mem_after-mem_before:+.2f})")
            print(f"  needle={'YES' if found else 'NO'}: {comp[:60]!r}")
            results.append(r)
        except Exception as e:
            print(f"  ✗ FAIL @ {ctx}: {e}")
            results.append({"ctx_target": ctx, "error": str(e)})

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"pr1_fp8kv_tp{args.tp}_{ts}.json"
    out_path.write_text(json.dumps({
        "stage": "PR1a-fp8kv",
        "model": MODEL_ID, "tp": args.tp, "max_ctx": args.max_ctx,
        "gmu": args.gmu, "kv_dtype": args.kv_dtype, "ts": ts,
        "results": results,
    }, indent=2))
    print(f"\n✓ saved {out_path}")
    print("\nSummary:")
    print(f"  {'ctx':>10}  {'mem GB':>8}  {'t/s':>8}  needle")
    for r in results:
        if "error" in r:
            print(f"  {r['ctx_target']:>10,}  {'FAIL':>8}  {'-':>8}  -")
        else:
            print(f"  {r['ctx_target']:>10,}  {r['mem_after_gb']:>8.2f}  {r['throughput_tok_s']:>8.0f}  {'✓' if r['needle_found'] else '✗'}")


if __name__ == "__main__":
    main()
