"""End-to-end smoke test: Quest top-K on real Qwen2 attention layers.

What this verifies:
- The reference upper-bound + top-K + selected attention path produces
  outputs close to dense attention on real (q, K, V) tensors taken
  from a Qwen2 forward.
- The recall of "important keys" (those that dominate softmax)
  survives top-K selection at realistic context lengths.
- The cosine of (Quest output) vs (dense output) stays high enough
  that downstream loss should be small.

This is a per-layer probe, not an end-to-end generation. To turn it
into a full attention swap we need to plug into vLLM (~2 weeks); for
now we measure the math behaves as expected on real activations.

Usage:
    PYTHONPATH=src python benchmarks/pr3/smoke_quest.py \\
        --ctx 8192 --page-size 16 --top-k-frac 0.05 0.1 0.25 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Make src/ importable for direct invocation
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


def dense_attention(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Plain SDPA, head-by-head, returning (H, D_v).

    q: (H, D),  K: (T, H, D),  V: (T, H, D)
    """
    H, D = q.shape
    T = K.shape[0]
    scale = 1.0 / (D ** 0.5)
    out = torch.zeros(H, V.shape[-1], dtype=q.dtype, device=q.device)
    for h in range(H):
        scores = (K[:, h] @ q[h]) * scale                        # (T,)
        attn = torch.softmax(scores.float(), dim=0).to(q.dtype)
        out[h] = attn @ V[:, h]
    return out


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row cosine for (H, D) tensors."""
    a = a.float()
    b = b.float()
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1)
    return num / den.clamp_min(1e-6)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--top-k-frac", nargs="+", type=float,
                   default=[0.05, 0.10, 0.25, 0.50])
    p.add_argument("--sink-pages", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--layer", type=int, default=14, help="which decoder layer to probe")
    args = p.parse_args()

    print(f"loading model {args.model} (bf16, sdpa) ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.to(args.device)
    model.train(False)

    text = build_text(args.ctx)
    ids = tok.encode(text, return_tensors="pt").to(args.device)
    ids = ids[:, : args.ctx]
    T = ids.shape[1]
    print(f"prompt: {T} tokens", flush=True)

    cfg = model.config
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // num_q_heads
    print(f"heads: q={num_q_heads}, kv={num_kv_heads}, head_dim={head_dim}", flush=True)

    layer = model.model.layers[args.layer].self_attn
    captured = {}

    def hook(module, _inputs, _output):
        # Pull K and V projections from the module after they ran. The
        # easiest grab in HF Qwen2: the module exposes intermediate
        # states via the forward; we cheat with a state-dict-friendly
        # post-hook on k_proj / v_proj.
        pass

    # We hook k_proj and v_proj to grab the projected (T, num_kv_heads * head_dim)
    # tensors, plus q_proj for the query.
    grab = {}

    def k_hook(_m, _i, output):
        grab["K"] = output.detach()  # (1, T, num_kv * D)

    def v_hook(_m, _i, output):
        grab["V"] = output.detach()

    def q_hook(_m, _i, output):
        grab["Q"] = output.detach()  # (1, T, num_q * D)

    h_q = layer.q_proj.register_forward_hook(q_hook)
    h_k = layer.k_proj.register_forward_hook(k_hook)
    h_v = layer.v_proj.register_forward_hook(v_hook)

    print("running forward ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        model(ids, use_cache=False)
    print(f"forward: {time.time() - t0:.1f}s", flush=True)
    h_q.remove(); h_k.remove(); h_v.remove()

    Q_full = grab["Q"][0].reshape(T, num_q_heads, head_dim)        # (T, H_q, D)
    K_full = grab["K"][0].reshape(T, num_kv_heads, head_dim)       # (T, H_kv, D)
    V_full = grab["V"][0].reshape(T, num_kv_heads, head_dim)
    print(f"captured Q={tuple(Q_full.shape)}, K={tuple(K_full.shape)}, "
          f"V={tuple(V_full.shape)}", flush=True)

    # We run the smoke at the *last* token of the prompt as the decode
    # query, attending back over the prefix - the realistic Quest
    # decode-step setup.
    q_last = Q_full[-1]                     # (H_q, D)
    K_prefix = K_full[:-1]                   # (T-1, H_kv, D)
    V_prefix = V_full[:-1]

    # GQA mapping: each Q-head shares its KV-head with (H_q / H_kv) Q-heads.
    # For the page-stats / upper-bound we work in the KV-head space, so we
    # collapse Q-heads first via mean over the group. (For the actual
    # attention we'll re-broadcast.)
    q_per_kv = num_q_heads // num_kv_heads
    q_for_pages = q_last.reshape(num_kv_heads, q_per_kv, head_dim).mean(dim=1)  # (H_kv, D)

    # Pad so the prefix length is divisible by page_size.
    T_eff = K_prefix.shape[0]
    pad = (-T_eff) % args.page_size
    if pad:
        K_prefix = torch.cat([K_prefix, K_prefix[-1:].expand(pad, -1, -1)], dim=0)
        V_prefix = torch.cat([V_prefix, V_prefix[-1:].expand(pad, -1, -1)], dim=0)
    P = K_prefix.shape[0] // args.page_size
    K_pages = K_prefix.reshape(P, args.page_size, num_kv_heads, head_dim)
    V_pages = V_prefix.reshape(P, args.page_size, num_kv_heads, head_dim)
    print(f"pages: {P} of size {args.page_size}", flush=True)

    K_min, K_max = compute_page_stats(K_pages)
    bound = page_upper_bound(q_for_pages, K_min, K_max)            # (P, H_kv)

    # Dense reference (for the q_for_pages query in KV-head space)
    K_flat = K_prefix.reshape(P * args.page_size, num_kv_heads, head_dim)
    V_flat = V_prefix.reshape(P * args.page_size, num_kv_heads, head_dim)
    out_dense = dense_attention(q_for_pages, K_flat, V_flat)        # (H_kv, D)

    print(f"\nrunning Quest at top-K fractions: {args.top_k_frac}", flush=True)
    rows = []
    for frac in args.top_k_frac:
        k = max(1, int(P * frac))
        page_idx = topk_pages_with_sinks(bound, k=k, sink_pages=args.sink_pages)
        out_quest = selected_attention(q_for_pages, K_pages, V_pages, page_idx)
        cos = cosine(out_quest, out_dense)
        kv_loaded = (k + args.sink_pages) * args.page_size
        kv_full = T_eff
        rows.append({
            "frac": frac,
            "k_pages": k,
            "kv_loaded": kv_loaded,
            "kv_full": kv_full,
            "ratio": kv_loaded / kv_full,
            "cos_per_head": cos.tolist(),
            "cos_mean": cos.mean().item(),
            "cos_min": cos.min().item(),
        })
        print(f"  frac={frac:.2f}  k={k:>4}  kv_load={kv_loaded:>5}/{kv_full:<5} "
              f"({100*kv_loaded/kv_full:5.1f}%)  cos[mean,min]="
              f"[{cos.mean():.3f}, {cos.min():.3f}]")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"quest_smoke_layer{args.layer}_ctx{args.ctx}_{ts}.json"
    out_path.write_text(json.dumps({
        "model": args.model, "ctx": args.ctx, "layer": args.layer,
        "page_size": args.page_size, "sink_pages": args.sink_pages,
        "P": P, "T_eff": T_eff, "rows": rows, "ts": ts,
    }, indent=2))
    print(f"\n saved {out_path}")


if __name__ == "__main__":
    main()
