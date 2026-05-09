"""Quest top-K quality across all 28 decoder layers.

Single forward pass on a fixed prefix, hooks every layer's q/k/v
projections, then for each layer runs Quest at a few top-K fractions
and records the cosine of (Quest output) vs (dense output).

This is the cross-layer counterpart to ``smoke_quest.py`` (which
tested one layer): we want to know if there is a "layer-27"-style
bad actor for Quest the way nuq2 had for Values, before we commit
to integrating Quest into vLLM.

Usage:
    PYTHONPATH=src python benchmarks/pr3/quest_layer_sweep.py \\
        --ctx 8192 --top-k-frac 0.05 0.10 0.25
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

from quest.reference import (
    compute_page_stats,
    page_upper_bound,
    selected_attention,
    topk_pages_with_sinks,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_text(target_tokens: int) -> str:
    return ("The river follows the valley to the sea, slow and silver in the dawn light. "
            "Birds wheel against the grey sky, calling. " * 200) * (target_tokens // 1000 + 1)


def dense_attention(q, K, V):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--top-k-frac", nargs="+", type=float, default=[0.05, 0.10, 0.25])
    p.add_argument("--sink-pages", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    print(f"loading {args.model} (bf16, sdpa) ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.to(args.device); model.train(False)

    cfg = model.config
    num_q = cfg.num_attention_heads
    num_kv = cfg.num_key_value_heads
    D = cfg.hidden_size // num_q
    layers = model.model.layers
    L = len(layers)
    print(f"layers={L}, q_heads={num_q}, kv_heads={num_kv}, head_dim={D}", flush=True)

    # Hook every layer's q_proj / k_proj / v_proj
    grab = [{} for _ in range(L)]
    handles = []
    def make_hook(idx, key):
        def hook(_m, _i, output):
            grab[idx][key] = output.detach()
        return hook
    for i, layer in enumerate(layers):
        handles.append(layer.self_attn.q_proj.register_forward_hook(make_hook(i, "Q")))
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(i, "K")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(i, "V")))

    text = build_text(args.ctx)
    ids = tok.encode(text, return_tensors="pt").to(args.device)[:, : args.ctx]
    T = ids.shape[1]
    print(f"prompt: {T} tokens. Running forward ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        model(ids, use_cache=False)
    print(f"forward: {time.time() - t0:.1f}s", flush=True)
    for h in handles:
        h.remove()

    # Pad seq to a multiple of page_size for the prefix [0, T-1)
    T_eff = T - 1
    pad = (-T_eff) % args.page_size
    P = (T_eff + pad) // args.page_size
    print(f"\npages: {P} of size {args.page_size} ({T_eff} prefix + {pad} pad)\n",
          flush=True)
    print(f"  {'layer':>5} | " + " | ".join(
        f"frac={f:.2f} cos[mean,min]" for f in args.top_k_frac))
    print("  " + "-" * 80)

    rows = []
    for i in range(L):
        Q_full = grab[i]["Q"][0].reshape(T, num_q, D)
        K_full = grab[i]["K"][0].reshape(T, num_kv, D)
        V_full = grab[i]["V"][0].reshape(T, num_kv, D)

        q_last = Q_full[-1]
        K_prefix = K_full[:T_eff]
        V_prefix = V_full[:T_eff]

        q_per_kv = num_q // num_kv
        q_for_pages = q_last.reshape(num_kv, q_per_kv, D).mean(dim=1)

        if pad:
            K_prefix = torch.cat([K_prefix, K_prefix[-1:].expand(pad, -1, -1)], dim=0)
            V_prefix = torch.cat([V_prefix, V_prefix[-1:].expand(pad, -1, -1)], dim=0)
        K_pages = K_prefix.reshape(P, args.page_size, num_kv, D)
        V_pages = V_prefix.reshape(P, args.page_size, num_kv, D)

        K_min, K_max = compute_page_stats(K_pages)
        bound = page_upper_bound(q_for_pages, K_min, K_max)

        K_flat = K_prefix.reshape(P * args.page_size, num_kv, D)
        V_flat = V_prefix.reshape(P * args.page_size, num_kv, D)
        out_dense = dense_attention(q_for_pages, K_flat, V_flat)

        layer_row = {"layer": i, "results": []}
        line = f"  {i:>5}  "
        for frac in args.top_k_frac:
            k = max(1, int(P * frac))
            page_idx = topk_pages_with_sinks(bound, k=k, sink_pages=args.sink_pages)
            out_quest = selected_attention(q_for_pages, K_pages, V_pages, page_idx)
            cos = cosine(out_quest, out_dense)
            cm, cmin = cos.mean().item(), cos.min().item()
            layer_row["results"].append({
                "frac": frac, "cos_mean": cm, "cos_min": cmin})
            line += f"|  [{cm:.3f}, {cmin:.3f}]      "
        rows.append(layer_row)
        print(line)
        # Free memory before next layer
        grab[i] = {}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"quest_layer_sweep_ctx{args.ctx}_{ts}.json"
    out_path.write_text(json.dumps({
        "model": args.model, "ctx": args.ctx, "page_size": args.page_size,
        "sink_pages": args.sink_pages, "P": P, "T_eff": T_eff,
        "top_k_frac": args.top_k_frac, "rows": rows, "ts": ts,
    }, indent=2))
    print(f"\n saved {out_path}")

    # Summary: which layers struggle?
    print("\n=== layers with cos_min < 0.7 at any top-K ===")
    for layer_row in rows:
        bad = [r for r in layer_row["results"] if r["cos_min"] < 0.7]
        if bad:
            for r in bad:
                print(f"  layer {layer_row['layer']:>2} frac={r['frac']:.2f}: "
                      f"cos_mean={r['cos_mean']:.3f}, cos_min={r['cos_min']:.3f}")


if __name__ == "__main__":
    main()
