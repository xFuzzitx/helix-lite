"""Multi-needle NIAH harness — *EM-RAG path*.

Same needle pool / depth distribution as ``run_multi_needle.py``,
but instead of stuffing the full document into the AWQ vanilla
context, it goes through the EM-RAG pipeline: segment + retrieve
top-M episodes per question + answer with AWQ.

This is the path that lets HELIX-Lite query documents larger than
the 128 K AWQ window — so it's the one that needs its own NIAH
score.

Two-process orchestration is used because CUDA state initialized by
the HF indexer poisons vLLM's engine-core spawn at scale.

  Phase A (this process): HF indexer
    1. Load HF once
    2. For each ctx: build index, retrieve top-M episode texts per
       question, accumulate to a per-question retrieval cache.
    3. Free HF, write the retrieval cache to JSON.

  Phase B (clean subprocess): vLLM AWQ
    4. Read the retrieval cache.
    5. Load vLLM AWQ once.
    6. For each cached (question, retrieved-context) pair: generate.
    7. Write per-question answers to JSON.

Phase A re-execs Phase B as a clean python child so CUDA is fresh.

Usage:
    PYTHONPATH=src python benchmarks/quality/run_em_rag_multi_needle.py \\
        --ctx 32000 128000 200000 --num-needles 8 --top-m 16
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.quality.run_multi_needle import build_prompt  # noqa: E402

RESULTS_DIR = ROOT / "benchmarks" / "results"
TMP_DIR = Path("/tmp")


def phase_a_retrieve(args) -> Path:
    """Build indices + retrieve, write retrieval JSON for phase B."""
    from helix.em_rag import (
        EMRAGConfig, _load_indexer, build_index, retrieve_episode_texts,
    )
    from transformers import AutoTokenizer

    em_cfg = EMRAGConfig(
        indexer_model=args.indexer,
        generator_model=args.generator,
        top_m=args.top_m,
        max_doc_tokens=args.max_doc_tokens,
        pool=args.pool,
        query_pool=args.query_pool,
        pool_alpha=args.pool_alpha,
        indexer_layer=args.indexer_layer,
        hyde=args.hyde,
        hyde_max_tokens=args.hyde_max_tokens,
        rerank=args.rerank,
        rerank_topk=args.rerank_topk,
        rerank_max_episode_chars=args.rerank_max_episode_chars,
    )
    grading_tok = AutoTokenizer.from_pretrained(args.generator)

    payload = {
        "indexer": args.indexer, "generator": args.generator,
        "top_m": args.top_m, "num_needles": args.num_needles,
        "seed": args.seed, "max_tokens": args.max_tokens,
        "pool": args.pool, "query_pool": args.query_pool,
        "pool_alpha": args.pool_alpha, "indexer_layer": args.indexer_layer,
        "hyde": args.hyde, "hyde_max_tokens": args.hyde_max_tokens,
        "rerank": args.rerank, "rerank_topk": args.rerank_topk,
        "ctxs": [],
    }

    print(f"=== Phase A: HF indexer ({args.indexer}) ===")
    t0 = time.time()
    model, tokenizer = _load_indexer(em_cfg)
    try:
        for ctx in args.ctx:
            print(f"\n--- ctx={ctx:,} ---")
            prompt, needles = build_prompt(ctx, args.num_needles, grading_tok,
                                            seed=args.seed)
            actual = len(grading_tok.encode(prompt))
            print(f"  prompt length: {len(prompt):,} chars / {actual:,} tok")
            t_idx0 = time.time()
            index = build_index(prompt, em_cfg, model=model, tokenizer=tokenizer)
            t_idx = time.time() - t_idx0
            print(f"  indexed: {index.store.num_episodes} episodes "
                  f"in {t_idx:.1f}s ({actual/t_idx:.0f} tok/s)")

            t_ret0 = time.time()
            retrieval_per_q = []
            for n in needles:
                texts, ranges = retrieve_episode_texts(
                    index, n["question"], em_cfg,
                    model=model, tokenizer=tokenizer)
                retrieval_per_q.append({
                    "question": n["question"],
                    "value": n["value"],
                    "depth": n["approx_depth"],
                    "retrieved_text": "\n\n---\n\n".join(texts),
                    "retrieved_chars": sum(len(t) for t in texts),
                })
            t_ret = time.time() - t_ret0
            print(f"  retrieved {args.num_needles} questions in {t_ret:.1f}s")

            payload["ctxs"].append({
                "ctx_target": ctx, "ctx_actual": actual,
                "num_episodes": index.store.num_episodes,
                "indexing_s": t_idx, "retrieval_s": t_ret,
                "questions": retrieval_per_q,
            })
            del index
    finally:
        del model
        torch.cuda.empty_cache()

    print(f"\nPhase A done in {time.time()-t0:.1f}s total")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    cache_path = TMP_DIR / f"em_rag_retrieval_{ts}.json"
    cache_path.write_text(json.dumps(payload, indent=2))
    print(f"✓ retrieval cache: {cache_path}")
    return cache_path


def phase_b_generate(cache_path: Path, gmu: float) -> Path:
    """Read retrieval cache, generate via vLLM AWQ, return per-q results."""
    payload = json.loads(cache_path.read_text())
    from helix.vanilla import VanillaConfig, VanillaSession

    print(f"=== Phase B: vLLM AWQ ({payload['generator']}) ===")
    t0 = time.time()
    session = VanillaSession(VanillaConfig(model=payload["generator"],
                                            gpu_memory_utilization=gmu))
    print(f"  loaded in {time.time()-t0:.1f}s")

    answers_payload = {**payload, "ctxs_with_answers": []}
    for ctx_block in payload["ctxs"]:
        ctx = ctx_block["ctx_target"]
        print(f"\n--- ctx={ctx:,} ---")
        t1 = time.time()
        per_q = []
        verdicts = []
        for q in ctx_block["questions"]:
            ans = session.query(question=q["question"],
                                 document=q["retrieved_text"],
                                 max_tokens=payload["max_tokens"],
                                 temperature=0.0)
            ok = q["value"] in ans
            verdicts.append(ok)
            per_q.append({**q, "found": ok, "answer_preview": ans[:300]})
            mark = "✓" if ok else "✗"
            print(f"  {mark} value={q['value']!r:>10} depth={q['depth']:.2f} "
                  f"retrieved={q['retrieved_chars']:,}ch")
        elapsed = time.time() - t1
        recall = sum(verdicts) / len(verdicts)
        print(f"  AWQ phase: {elapsed:.1f}s  "
              f"recall={sum(verdicts)}/{len(verdicts)} ({100*recall:.0f}%)")
        answers_payload["ctxs_with_answers"].append({
            **{k: v for k, v in ctx_block.items() if k != "questions"},
            "generation_s": elapsed,
            "recall": recall, "found": sum(verdicts), "total": len(verdicts),
            "questions": per_q,
        })

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_topm = payload["top_m"]
    pool_tag = payload.get("pool", "max-abs").replace("+", "_")
    layer_tag = payload.get("indexer_layer", "last")
    hyde_tag = "_hyde" if payload.get("hyde") else ""
    rerank_tag = f"_rerank{payload.get('rerank_topk', 0)}" if payload.get("rerank") else ""
    out_path = (
        RESULTS_DIR
        / f"em_rag_multi_needle_topm{safe_topm}_pool-{pool_tag}_layer-{layer_tag}{hyde_tag}{rerank_tag}_{ts}.json"
    )
    answers_payload["ts"] = ts
    out_path.write_text(json.dumps(answers_payload, indent=2))
    print(f"\n✓ saved {out_path}")
    print("\nSummary:")
    print(f"  {'ctx':>10}  recall    eps  index  retr   gen")
    for r in answers_payload["ctxs_with_answers"]:
        print(f"  {r['ctx_target']:>10,}  {r['found']}/{r['total']:>2}  "
              f"{r['num_episodes']:>4}  {r['indexing_s']:>5.1f}s "
              f"{r['retrieval_s']:>4.1f}s {r['generation_s']:>5.1f}s")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--indexer", default="Qwen/Qwen2.5-7B-Instruct-1M")
    p.add_argument("--generator", default="graelo/Qwen2.5-7B-Instruct-1M-AWQ")
    p.add_argument("--ctx", nargs="+", type=int, default=[32_000, 128_000])
    p.add_argument("--num-needles", type=int, default=8)
    p.add_argument("--top-m", type=int, default=16)
    p.add_argument("--max-doc-tokens", type=int, default=200_000)
    p.add_argument("--gmu", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--pool", default="max-abs",
                   choices=["max-abs", "mean", "last", "mean+max"],
                   help="episode embedding pooling (default: max-abs)")
    p.add_argument("--query-pool", default="last",
                   choices=["max-abs", "mean", "last", "mean+max"],
                   help="question embedding pooling (default: last)")
    p.add_argument("--pool-alpha", type=float, default=0.5,
                   help="weight for max-abs term in 'mean+max' pool")
    p.add_argument("--indexer-layer", default="mid",
                   choices=["last", "mid", "multi-last4", "multi-mid4"],
                   help="which layer's hidden state to use (default: mid)")
    p.add_argument("--hyde", action="store_true",
                   help="enable HyDE query expansion (generate hypothetical "
                        "passage + embed that instead of bare question)")
    p.add_argument("--hyde-max-tokens", type=int, default=48,
                   help="max tokens of hypothetical-passage continuation")
    p.add_argument("--rerank", action="store_true",
                   help="enable LLM-as-reranker on top of cosine retrieval")
    p.add_argument("--rerank-topk", type=int, default=256,
                   help="cosine top-K before rerank (default: 256)")
    p.add_argument("--rerank-max-episode-chars", type=int, default=2000,
                   help="truncate long episodes for the rerank prompt")
    p.add_argument("--phase", choices=["all", "a", "b"], default="all",
                   help="'all' = run A then re-exec B in subprocess; "
                        "'a' = only HF indexing+retrieval; "
                        "'b' = only AWQ generation (needs --cache)")
    p.add_argument("--cache", type=Path, help="retrieval cache JSON for phase b")
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase == "b":
        if args.cache is None or not args.cache.exists():
            sys.exit("--phase b requires --cache <path>")
        phase_b_generate(args.cache, args.gmu)
        return

    cache_path = phase_a_retrieve(args)
    if args.phase == "a":
        return

    # Phase B — re-exec ourselves in a clean subprocess
    print("\n→ launching clean subprocess for Phase B (vLLM AWQ) ...\n", flush=True)
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--phase", "b", "--cache", str(cache_path),
           "--gmu", str(args.gmu)]
    rc = subprocess.call(cmd, env=os.environ)
    if rc != 0:
        sys.exit(f"Phase B subprocess failed with rc={rc}")


if __name__ == "__main__":
    main()
