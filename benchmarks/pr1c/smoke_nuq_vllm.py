"""Phase 1A smoke for nuq KV in vLLM.

Loads ``graelo/Qwen2.5-7B-Instruct-1M-AWQ`` with the FA backend
monkey-patched to route through ``KVQuantAttentionImpl``, then runs
a tiny NIAH at increasing context lengths to confirm the needle
survives the math-wrapper.

Pass criterion: the model's answer contains the needle's value
substring at the smoke ctx. If yes, the calibrated scales survive
vLLM's prefill+decode pipeline end-to-end and we can move on to
Phase 1B (real compact pool).

Usage::

    PYTHONPATH=src python benchmarks/pr1c/smoke_nuq_vllm.py \\
        --scales scales/qwen2_5_7b_1m_nuq4_v3.pt --ctx 4000 32000
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scales", required=True, type=Path,
                   help="path to KVScales .pt file")
    p.add_argument("--model", default="graelo/Qwen2.5-7B-Instruct-1M-AWQ")
    p.add_argument("--ctx", nargs="+", type=int, default=[4000, 32000])
    p.add_argument("--max-model-len", type=int, default=128_000)
    p.add_argument("--gmu", type=float, default=0.85)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--baseline", action="store_true",
                   help="skip the nuq install (= vanilla AWQ, for A/B)")
    p.add_argument("--compact", action="store_true",
                   help="use Phase 1B v1 compact pool (writes mirrored to "
                        "user-owned uint8 pool; FA still reads from fp16). "
                        "Validates pack-on-write integration end-to-end.")
    p.add_argument("--compact-num-blocks", type=int, default=20_000,
                   help="num_blocks for the compact pool (each = block_size tokens)")
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not args.scales.exists():
        sys.exit(f"scales file not found: {args.scales}")

    if args.baseline:
        print("[smoke] BASELINE — no kvquant, vanilla FA")
    elif args.compact:
        from kvquant.compact_backend import install_compact_backend
        install_compact_backend(
            str(args.scales),
            num_layers=28, num_blocks=args.compact_num_blocks,
            block_size=16, num_kv_heads=4, head_size=128,
            device=args.device, top_k_blocks_decode=0,
        )
        print("[smoke] PR1c Phase 1B v1 compact backend installed")
    else:
        from kvquant.vllm_backend import install_kvquant_backend
        install_kvquant_backend(str(args.scales), device=args.device)
        print("[smoke] PR1c Phase 1A math-wrapper backend installed")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"[smoke] loading {args.model} ...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype="auto",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gmu,
        enforce_eager=True,
        tensor_parallel_size=1,
    )
    print(f"[smoke] loaded in {time.time()-t0:.1f}s")
    tok = AutoTokenizer.from_pretrained(args.model)

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, top_p=1.0)

    rng = random.Random(args.seed)
    filler = "The river follows the valley to the sea, slow and silver in the dawn light. " * 60 + "\n\n"

    overall_ok = True
    for ctx in args.ctx:
        needle_value = f"BANANA-{rng.randrange(1000, 9999)}"
        needle = f"The secret password is {needle_value}."
        depth = 0.5  # middle of context
        chars_total = max(0, ctx * 4 - len(needle) - 400)
        body = filler * (chars_total // len(filler) + 1)
        body = body[:chars_total]
        cut = int(len(body) * depth)
        prompt = (
            body[:cut]
            + f"\n\n{needle}\n\n"
            + body[cut:]
            + "\n\n---\nQuestion: What is the secret password? "
              "Answer with the password only.\nAnswer:"
        )
        actual_tok = len(tok.encode(prompt))
        print(f"\n--- ctx={ctx:,} (actual {actual_tok:,} tok), needle={needle_value} ---")
        t0 = time.time()
        out = llm.generate([prompt], sampling)
        elapsed = time.time() - t0
        ans = out[0].outputs[0].text
        ok = needle_value in ans
        mark = "✓" if ok else "✗"
        print(f"  {mark} elapsed={elapsed:.1f}s  answer={ans.strip()[:160]!r}")
        overall_ok = overall_ok and ok

    print(f"\n{'=== PASS ===' if overall_ok else '=== FAIL ==='}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
