"""Retrieval-correctness test for EM-LLM on NIAH.

Question: in a long NIAH-style prompt, does the embedding-based
top-M retrieval surface the episode that contains the needle?

If yes, the retrieval level of the EM-LLM chain works - the cold
episodes a model would consult at decode time would include the
right one. This is independent of the attention swap math (covered
by smoke_em_swap.py) and of generation correctness (covered by the
multi-needle eval on AWQ).

Method:
    1. Construct a NIAH prompt with N distinct (key, value) pairs at
       known depths.
    2. Forward through Qwen2 in bf16, capture last-layer hidden
       states and per-layer KV.
    3. Segment with Bayesian surprise.
    4. Build a query "what is the value of <key>?" - tokenise and
       run that *as a separate prompt* through the model to get a
       proper hidden state for the question (not last-prompt-token,
       which is biased by recency).
    5. For each (key, value, depth), retrieve top-M episodes by
       cosine and check whether the needle's depth falls in any of
       them.

The score is mean recall over the N needles: (needles found in
top-M) / N.

Usage:
    PYTHONPATH=src python benchmarks/pr5/needle_retrieval.py \\
        --ctx 32000 --num-needles 8 --top-m 4 8 16
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from emllm.kv_store import KVEpisodeStore
from emllm.segmenter import BayesianSurpriseSegmenter, SegmenterConfig


RESULTS_DIR = Path(__file__).resolve().parent / "results"


# Needle templates: (statement template, question template)
NEEDLE_POOL = [
    ("The tropical fruit code for Bahrain is {value}.",
     "What is the tropical fruit code for Bahrain?"),
    ("The desert wind speed in Tatooine measures {value} kph.",
     "What is the desert wind speed in Tatooine?"),
    ("The migration corridor for swallows is highway {value}.",
     "Which highway is the migration corridor for swallows?"),
    ("The encryption salt for project Orion is {value}.",
     "What is the encryption salt for project Orion?"),
    ("The harvest yield in plot 47 reached {value} kilograms.",
     "What was the harvest yield in plot 47?"),
    ("The submarine call sign of the Nautilus is {value}.",
     "What is the submarine call sign of the Nautilus?"),
    ("The library shelf for forbidden manuscripts is {value}.",
     "Which shelf holds the forbidden manuscripts?"),
    ("The patron color of the spring festival is {value}.",
     "What is the patron color of the spring festival?"),
]


def make_value(rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return f"{rng.choice(letters)}{rng.choice(letters)}-{rng.randrange(1000, 9999)}"


def build_niah(target_chars: int, num_needles: int, seed: int = 0):
    rng = random.Random(seed)
    pool = list(NEEDLE_POOL)
    rng.shuffle(pool)
    pool = pool[:num_needles]
    needles = []
    for tpl, q in pool:
        v = make_value(rng)
        needles.append({"template": tpl, "question": q, "value": v,
                         "filled": tpl.format(value=v)})

    filler_unit = "The river follows the valley to the sea, slow and silver in the dawn light. " * 60 + "\n\n"
    body_chars = target_chars - sum(len(n["filled"]) + 4 for n in needles) - 200
    base = (filler_unit * (body_chars // len(filler_unit) + 1))[: max(body_chars, 0)]

    n = len(needles)
    L = len(base)
    positions = sorted(int(L * (i + 1) / (n + 1) + rng.uniform(-L * 0.04, L * 0.04))
                       for i in range(n))
    parts, last = [], 0
    for pos, needle in zip(positions, needles):
        parts.append(base[last:pos])
        parts.append(f"\n\n{needle['filled']}\n\n")
        last = pos
        needle["char_pos"] = pos
        needle["depth"] = pos / L
    parts.append(base[last:])
    return "".join(parts), needles


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    p.add_argument("--ctx", type=int, default=16384,
                   help="approximate token count of the prefix")
    p.add_argument("--num-needles", type=int, default=8)
    p.add_argument("--top-m", nargs="+", type=int, default=[4, 8, 16])
    p.add_argument("--threshold-quantile", type=float, default=0.95)
    p.add_argument("--min-seg", type=int, default=128)
    p.add_argument("--max-seg", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--store-device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ep-pool", default="mean", choices=["mean", "last", "max-abs"],
                   help="how to pool last-layer hidden states into one episode embedding")
    p.add_argument("--query-pool", default="last", choices=["mean", "last"],
                   help="how to pool the question's hidden states")
    args = p.parse_args()

    print(f"loading {args.model} (bf16, sdpa) ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.to(args.device)
    model.train(False)

    text, needles = build_niah(args.ctx * 4, args.num_needles, seed=args.seed)
    ids = tok.encode(text, return_tensors="pt").to(args.device)
    T = ids.shape[1]
    print(f"prompt: {T} tokens, {args.num_needles} needles", flush=True)

    # Compute approximate token positions for each needle from the
    # char positions, then verify by re-tokenising the prefix up to
    # each char_pos.
    for n in needles:
        prefix_text = text[: n["char_pos"]]
        n["token_pos"] = len(tok.encode(prefix_text))

    print("running forward (output_hidden_states=True) ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
    print(f"  forward: {time.time() - t0:.1f}s", flush=True)
    last_hidden = out.hidden_states[-1][0]
    logits = out.logits[0]

    seg_cfg = SegmenterConfig(
        threshold_quantile=args.threshold_quantile,
        window=128,
        min_segment_len=args.min_seg,
        max_segment_len=args.max_seg,
    )
    seg = BayesianSurpriseSegmenter(seg_cfg)
    boundaries: list[int] = []
    for t in range(T):
        b = seg.step(logits[t])
        if b is not None:
            boundaries.append(b.position)
    if not boundaries or boundaries[-1] < T:
        boundaries.append(T)
    print(f"\n{len(boundaries)} episodes built", flush=True)

    hidden_size = model.config.hidden_size
    store = KVEpisodeStore(emb_dim=hidden_size, device=args.store_device,
                           capacity=4096, dtype=torch.bfloat16)

    def pool(slab, mode):
        if mode == "mean":
            return slab.mean(dim=0)
        if mode == "last":
            return slab[-1]
        if mode == "max-abs":
            # Pick the per-channel value with largest magnitude across positions
            idx = slab.abs().argmax(dim=0)
            return slab.gather(0, idx.unsqueeze(0)).squeeze(0)
        raise ValueError(f"unknown pool mode {mode!r}")

    seg_start = 0
    episode_ranges = []
    for b in boundaries:
        emb = pool(last_hidden[seg_start:b], args.ep_pool)
        ep = store.add(emb, (seg_start, b))
        episode_ranges.append((seg_start, b))
        # We don't need KV for retrieval-only; skip add_kv to save GPU 1 RAM
        seg_start = b
    print(f"  {store!r}", flush=True)

    # Locate which episode each needle's token position falls in
    for n in needles:
        for ep_idx, (a, c) in enumerate(episode_ranges):
            if a <= n["token_pos"] < c:
                n["episode_idx"] = ep_idx
                break
        else:
            n["episode_idx"] = -1

    # For each needle, build a query-only forward to get a clean hidden
    # state representing the question, then top-M retrieve.
    print(f"\nrunning {args.num_needles} retrieval queries", flush=True)
    rows = []
    for top_m in args.top_m:
        recalls = []
        per_needle = []
        for n in needles:
            q_text = n["question"]
            q_ids = tok.encode(q_text, return_tensors="pt").to(args.device)
            with torch.no_grad():
                q_out = model(q_ids, output_hidden_states=True, use_cache=False)
            q_emb = pool(q_out.hidden_states[-1][0], args.query_pool)
            top = store.topk(q_emb, k=top_m)
            top_idx = [ep.index for ep, _ in top]
            hit = n["episode_idx"] in top_idx
            recalls.append(hit)
            per_needle.append({
                "value": n["value"], "depth": n["depth"],
                "token_pos": n["token_pos"],
                "episode_idx": n["episode_idx"],
                "top_idx": top_idx, "hit": hit,
            })
        recall = sum(recalls) / len(recalls)
        rows.append({"top_m": top_m, "recall": recall, "per_needle": per_needle})
        print(f"  top-M={top_m:>2}  recall={sum(recalls)}/{len(recalls)} "
              f"({100 * recall:.0f}%)")
        for r in per_needle:
            mark = "✓" if r["hit"] else "✗"
            print(f"    {mark} value={r['value']!r:>9}  depth={r['depth']:.2f}  "
                  f"ep={r['episode_idx']:>3}  top={r['top_idx']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"needle_retrieval_ctx{T}_{ts}.json"
    out_path.write_text(json.dumps({
        "model": args.model, "T": T, "num_episodes": store.num_episodes,
        "num_needles": args.num_needles, "rows": rows, "ts": ts,
    }, indent=2))
    print(f"\n saved {out_path}")


if __name__ == "__main__":
    main()
