"""End-to-end smoke test: stream a real document through Qwen2 and
populate an EpisodeStore.

What this verifies:
- The segmenter does not over- or under-fire on real Qwen logits.
- Hidden states extracted at boundaries produce embeddings that are
  meaningful enough for cosine retrieval to recover plausible
  neighbours.
- The cross-device flow works: model + logits + hidden states on
  GPU 0, episode pool on GPU 1.

Usage:
    PYTHONPATH=src python benchmarks/pr5/smoke_segmenter.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Ensure src/ is on PYTHONPATH for direct invocation
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from emllm.episode_store import EpisodeStore
from emllm.segmenter import BayesianSurpriseSegmenter, SegmenterConfig


def build_topic_mixed_text() -> str:
    """A synthetic but realistic stream with topic shifts roughly every
    400-500 tokens, so we can sanity-check that the segmenter doesn't
    miss the shifts."""
    return "\n\n".join(
        [
            "## Botany\n\n"
            + "Photosynthesis is the process by which green plants and certain other organisms transform light energy into chemical energy. "
            * 30,
            "## Number theory\n\n"
            + "A prime number is a natural number greater than 1 that is not a product of two smaller natural numbers. "
            * 30,
            "## Music history\n\n"
            + "Johann Sebastian Bach was a German composer of the Baroque era, regarded as one of the greatest composers of all time. "
            * 30,
            "## Cooking\n\n"
            + "To make a basic vinaigrette, whisk three parts oil with one part vinegar, mustard, salt, and pepper until emulsified. "
            * 30,
            "## Physics\n\n"
            + "Quantum entanglement is a phenomenon where two or more particles become correlated in such a way that the quantum state of each cannot be described independently. "
            * 30,
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--store-device", default=None,
                   help="default cuda:1 if available else cuda:0")
    p.add_argument("--threshold-quantile", type=float, default=0.92)
    p.add_argument("--min-seg", type=int, default=64)
    p.add_argument("--max-seg", type=int, default=512)
    args = p.parse_args()

    print("loading model ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    # bf16 instead of fp16: Qwen2.5-7B-1M produces NaN hidden states in
    # fp16 + eager attention because residual-stream magnitudes overflow
    # fp16's dynamic range. bf16 has the same exponent as fp32, so no
    # overflow at the cost of mantissa precision (which doesn't matter
    # for retrieval).
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.to(args.device)
    model.train(False)

    text = build_topic_mixed_text()
    ids = tok.encode(text, return_tensors="pt").to(args.device)
    T = ids.shape[1]
    print(f"streamed prompt: {T} tokens, {len(text)} chars", flush=True)

    cfg = SegmenterConfig(
        threshold_quantile=args.threshold_quantile,
        window=128,
        min_segment_len=args.min_seg,
        max_segment_len=args.max_seg,
    )
    seg = BayesianSurpriseSegmenter(cfg)
    store = EpisodeStore(
        emb_dim=model.config.hidden_size,
        device=args.store_device,
        capacity=512,
    )
    print(f"store initialised: {store!r}", flush=True)

    t0 = time.time()
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
    print(f"forward pass: {time.time() - t0:.1f}s", flush=True)

    last_hidden = out.hidden_states[-1][0]    # (T, D)
    logits = out.logits[0]                    # (T, vocab)

    seg_start = 0
    boundaries: list[int] = []
    for t in range(T):
        b = seg.step(logits[t])
        if b is not None:
            boundaries.append(b.position)
            embedding = last_hidden[seg_start : b.position].mean(dim=0)
            store.add(embedding, (seg_start, b.position), b.surprise)
            seg_start = b.position
    # close out the tail
    if seg_start < T:
        embedding = last_hidden[seg_start:T].mean(dim=0)
        store.add(embedding, (seg_start, T), 0.0)
        seg.close()
        boundaries.append(T)

    print(f"\ndetected {len(boundaries)} boundaries:")
    for i, b in enumerate(boundaries):
        prev = boundaries[i - 1] if i > 0 else 0
        decoded = tok.decode(ids[0, prev : prev + 12]).strip().replace("\n", " ")
        print(f"  ep[{i:>2}] tokens [{prev:>4}, {b:>4})  starts: {decoded[:60]!r}")

    # Retrieval sanity check: sample one position from each detected
    # episode and query with its hidden state - top-1 should usually
    # be the episode that contains it.
    print(f"\n{store!r}")
    print("\nself-recall (query = hidden state at episode midpoint):")
    correct = 0
    for ep in store.episodes:
        mid = (ep.token_range[0] + ep.token_range[1]) // 2
        q = last_hidden[mid]
        top = store.topk(q, k=1)
        if top and top[0][0].index == ep.index:
            correct += 1
    rate = correct / max(1, store.num_episodes)
    print(f"  top-1 self-recall: {correct}/{store.num_episodes} ({100 * rate:.0f}%)")

    # Cross-topic retrieval: take a query from the middle of episode 0 (botany)
    # and report the top-3 - we expect ep 0 first, with the rest being other
    # botany-ish or noise episodes (which there are none of in this synthetic
    # stream, so the assertion is loose).
    if store.num_episodes >= 3:
        ep0 = store.episodes[0]
        q = last_hidden[(ep0.token_range[0] + ep0.token_range[1]) // 2]
        top = store.topk(q, k=3)
        print("\nquery from ep[0] -> top 3:")
        for ep, score in top:
            print(f"  ep[{ep.index}] tokens [{ep.token_range[0]}, {ep.token_range[1]}) "
                  f"score={score:.3f}")


if __name__ == "__main__":
    main()
