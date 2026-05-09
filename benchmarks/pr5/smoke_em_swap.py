"""End-to-end smoke for the EM-LLM hot/cold attention swap.

What this proves: at decode time, attention over

    [sinks ; retrieved-cold-episodes ; hot-window]

approximates dense attention over the full prefix well enough that
quality is preserved - the architectural payoff that makes 5M
effective context tractable.

Method:
    1. Run a long (16K) prompt through Qwen2.5-7B-1M, capture
       (Q, K, V) at every layer.
    2. Segment via Bayesian surprise on the per-step logits.
    3. For each detected episode, build:
       - an embedding (mean of last-layer hidden over the span)
       - a KV chunk (per-layer K, V over the span) on cuda:1.
    4. For a chosen query position (the last token), compute:
       - dense_out: scaled-dot-product attention over the full prefix
       - swap_out: SDPA over assemble_kv(sinks, top-M cold, hot)
    5. Per-layer cosine of swap_out vs dense_out.

The (hot_window, top_m) sweep tells us how much of the cold context
a single query needs to attend to in order to track the dense
result. Tighter is better - it's the compression ratio.

Usage:
    PYTHONPATH=src python benchmarks/pr5/smoke_em_swap.py \\
        --ctx 16384 --hot-window 2048 4096 --top-m 4 8 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from emllm.episode_store import Episode
from emllm.kv_store import KVEpisodeStore
from emllm.hot_swap import HotSwapConfig, assemble_kv
from emllm.segmenter import BayesianSurpriseSegmenter, SegmenterConfig


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_text(target_tokens: int) -> str:
    """Topic-mixed stream so segmenter has signal to find boundaries."""
    parts = [
        "## Botany. Photosynthesis converts light into chemical energy in chloroplasts. " * 50,
        "## Astronomy. Black holes warp spacetime so severely that no light escapes the horizon. " * 50,
        "## Music. The sonata form develops two themes through exposition development recapitulation. " * 50,
        "## Cooking. To temper chocolate, melt it to 45C, cool to 27C, then warm to 31C. " * 50,
        "## Crypto. The discrete logarithm problem underlies most public-key cryptography. " * 50,
        "## History. The Treaty of Westphalia in 1648 ended the Thirty Years' War. " * 50,
        "## Geology. Plate tectonics explains earthquakes ridges and continental drift. " * 50,
        "## Linguistics. The Sapir-Whorf hypothesis claims language shapes cognition. " * 50,
    ]
    text = "\n\n".join(parts)
    while len(text) < target_tokens * 4:
        text += "\n\n" + "\n\n".join(parts)
    return text


def dense_attention(q, K, V):
    """SDPA per head; q (H, D), K (T, H, D), V (T, H, D) -> (H, D)."""
    H, D = q.shape
    scale = 1.0 / (D ** 0.5)
    out = torch.zeros(H, V.shape[-1], dtype=q.dtype, device=q.device)
    for h in range(H):
        scores = (K[:, h] @ q[h]) * scale
        attn = torch.softmax(scores.float(), dim=0).to(q.dtype)
        out[h] = attn @ V[:, h]
    return out


def cosine(a, b):
    a, b = a.float(), b.float()
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1)
    return num / den.clamp_min(1e-6)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--hot-window", nargs="+", type=int, default=[2048, 4096])
    p.add_argument("--top-m", nargs="+", type=int, default=[4, 8, 16])
    p.add_argument("--sink-tokens", type=int, default=4)
    p.add_argument("--threshold-quantile", type=float, default=0.95)
    p.add_argument("--min-seg", type=int, default=128)
    p.add_argument("--max-seg", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--store-device", default=None)
    p.add_argument("--probe-layers", nargs="+", type=int, default=None,
                   help="layers to compute cos similarity on; default = a "
                        "spread across the stack")
    args = p.parse_args()

    print(f"loading {args.model} (bf16, sdpa) ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.to(args.device)
    model.train(False)

    cfg = model.config
    num_q = cfg.num_attention_heads
    num_kv = cfg.num_key_value_heads
    D = cfg.hidden_size // num_q
    L = len(model.model.layers)
    if args.probe_layers is None:
        args.probe_layers = [0, L // 4, L // 2, (3 * L) // 4, L - 1]
    print(f"layers={L}, q_heads={num_q}, kv_heads={num_kv}, head_dim={D}", flush=True)

    text = build_text(args.ctx)
    ids = tok.encode(text, return_tensors="pt").to(args.device)[:, : args.ctx]
    T = ids.shape[1]
    print(f"prompt: {T} tokens", flush=True)

    grab = [{} for _ in range(L)]
    handles = []
    def make_hook(idx, key):
        def hook(_m, _i, output):
            grab[idx][key] = output.detach()
        return hook
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.q_proj.register_forward_hook(make_hook(i, "Q")))
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(i, "K")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(i, "V")))

    print("running forward (output_hidden_states=True for embeddings) ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
    print(f"forward: {time.time() - t0:.1f}s", flush=True)
    for h in handles:
        h.remove()

    last_hidden = out.hidden_states[-1][0]                        # (T, hidden)
    logits = out.logits[0]                                        # (T, vocab)

    # Stack the per-layer K, V into (L, T, H_kv, D) tensors
    K_all = torch.stack([grab[i]["K"][0].reshape(T, num_kv, D) for i in range(L)], dim=0)
    V_all = torch.stack([grab[i]["V"][0].reshape(T, num_kv, D) for i in range(L)], dim=0)
    Q_all = torch.stack([grab[i]["Q"][0].reshape(T, num_q, D) for i in range(L)], dim=0)
    print(f"K_all: {tuple(K_all.shape)}", flush=True)
    grab.clear()

    # Segment from logits
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

    print(f"\nbuilt {len(boundaries)} episodes", flush=True)

    # Build the KV episode store on GPU 1 (or wherever). Match the
    # model's bf16 dtype so we don't pay a cast on every retrieve.
    hidden_size = cfg.hidden_size
    store = KVEpisodeStore(emb_dim=hidden_size, device=args.store_device,
                           capacity=4096, dtype=torch.bfloat16)
    seg_start = 0
    for b in boundaries:
        # Embedding from the mean of the last-layer hidden over [seg_start, b)
        emb = last_hidden[seg_start:b].mean(dim=0)
        ep = store.add(emb, (seg_start, b))
        # KV chunk for this episode: (L, T_episode, H_kv, D)
        K_chunk = K_all[:, seg_start:b]
        V_chunk = V_all[:, seg_start:b]
        store.add_kv(ep, K_chunk, V_chunk)
        seg_start = b
    print(f"  {store!r}", flush=True)

    # The probe query: the very last token of the prompt (would be the
    # decode-time query in real generation).
    q_probe_idx = T - 1
    query_emb = last_hidden[q_probe_idx]                          # (hidden,)

    # Sinks (always-include first sink_tokens).
    sink_K = K_all[:, : args.sink_tokens]
    sink_V = V_all[:, : args.sink_tokens]

    # Map Q-heads -> KV-heads (GQA); we'll average q over each kv-group
    # for the embedding-based scoring while keeping the full Q for SDPA.
    q_per_kv = num_q // num_kv

    # The probe layer's full Q at the last position
    rows = []
    print(f"\nprobing layers: {args.probe_layers}", flush=True)

    # For each layer, compute dense_out once
    dense_per_layer: dict[int, torch.Tensor] = {}
    for layer_i in args.probe_layers:
        q_layer = Q_all[layer_i, q_probe_idx]                      # (num_q, D)
        K_layer_full = K_all[layer_i]                              # (T, num_kv, D)
        V_layer_full = V_all[layer_i]
        # Broadcast each kv head to its q heads to do full attention
        K_broad = K_layer_full.repeat_interleave(q_per_kv, dim=1)  # (T, num_q, D)
        V_broad = V_layer_full.repeat_interleave(q_per_kv, dim=1)
        dense_out = dense_attention(q_layer, K_broad, V_broad)     # (num_q, D)
        dense_per_layer[layer_i] = dense_out

    # Sweep
    print(f"\n{'hot':>5} {'M':>3} {'tokens':>10}  per-layer cos[mean]")
    print("  " + "-" * 78)
    for hot_window in args.hot_window:
        # The hot KV is the last hot_window tokens
        hot_K_all = K_all[:, max(args.sink_tokens, T - hot_window): T]
        hot_V_all = V_all[:, max(args.sink_tokens, T - hot_window): T]
        for top_m in args.top_m:
            cfg_swap = HotSwapConfig(
                hot_window=hot_window, top_m=top_m,
                sink_tokens=args.sink_tokens, metric="cosine",
            )
            # Filter cold candidates: only episodes that DON'T overlap the
            # hot region or sinks. Otherwise we'd duplicate tokens.
            cold_threshold = max(args.sink_tokens, T - hot_window)
            allowed = [i for i, ep in enumerate(store.episodes)
                       if ep.token_range[1] <= cold_threshold]
            if not allowed:
                # nothing cold to retrieve
                row = {"hot": hot_window, "top_m": top_m, "n_cold_avail": 0}
                rows.append(row)
                continue

            # Restrict topk to allowed (hack: zero out non-allowed embeddings
            # for scoring). Simpler: compute scores over all, mask.
            q_emb_dev = query_emb.to(store.device).float()
            active = store.embeddings[: store.num_episodes].float()
            q_norm = q_emb_dev.norm().clamp_min(1e-6)
            ep_norms = store.norms[: store.num_episodes].float().clamp_min(1e-6)
            scores = (active @ q_emb_dev) / (ep_norms * q_norm)
            mask = torch.full_like(scores, float("-inf"))
            mask_idx = torch.tensor(allowed, device=scores.device)
            mask[mask_idx] = 0.0
            scores_masked = scores + mask
            k = min(top_m, len(allowed))
            top = torch.topk(scores_masked, k=k, dim=0)
            top_idx = [int(top.indices[i].item()) for i in range(k)]
            # Sort by token range for causal coherence
            top_idx.sort(key=lambda i: store.episodes[i].token_range[0])

            cold_K, cold_V, _lens = store.gather_kv_for_episodes(top_idx, args.device)
            cold_len = cold_K.shape[1]

            line_cos = []
            for layer_i in args.probe_layers:
                # Build (sinks ; cold ; hot) for this layer
                K_combined = torch.cat([
                    sink_K[layer_i].to(args.device),
                    cold_K[layer_i],
                    hot_K_all[layer_i].to(args.device),
                ], dim=0)
                V_combined = torch.cat([
                    sink_V[layer_i].to(args.device),
                    cold_V[layer_i],
                    hot_V_all[layer_i].to(args.device),
                ], dim=0)
                # Broadcast to query heads
                K_broad = K_combined.repeat_interleave(q_per_kv, dim=1)
                V_broad = V_combined.repeat_interleave(q_per_kv, dim=1)
                q_layer = Q_all[layer_i, q_probe_idx]
                swap_out = dense_attention(q_layer, K_broad, V_broad)
                cos = cosine(swap_out, dense_per_layer[layer_i])
                line_cos.append((layer_i, cos.mean().item(), cos.min().item()))

            tokens = args.sink_tokens + cold_len + hot_K_all.shape[1]
            print(f"  {hot_window:>5} {top_m:>3} {tokens:>10,}  " +
                  "  ".join(f"L{li}:{cm:.3f}" for li, cm, _ in line_cos))
            rows.append({
                "hot": hot_window, "top_m": top_m,
                "tokens_seen": tokens, "tokens_full": T,
                "ratio": tokens / T,
                "per_layer": [{"layer": li, "cos_mean": cm, "cos_min": cmin}
                              for li, cm, cmin in line_cos],
            })

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"em_swap_ctx{args.ctx}_{ts}.json"
    out_path.write_text(json.dumps({
        "model": args.model, "ctx": args.ctx, "T": T,
        "num_episodes": store.num_episodes,
        "probe_layers": args.probe_layers,
        "hot_window": args.hot_window, "top_m": args.top_m,
        "sink_tokens": args.sink_tokens,
        "rows": rows, "ts": ts,
    }, indent=2))
    print(f"\n saved {out_path}")


if __name__ == "__main__":
    main()
