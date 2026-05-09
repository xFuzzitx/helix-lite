"""End-to-end validation of calibrated scales on a real Qwen2 forward pass.

What this does:
  1. Load a (small) Qwen2.5-7B-Instruct-1M in fp16.
  2. Hook ``k_proj`` / ``v_proj`` to capture activations on a NIAH prompt.
  3. Apply nuq2 pack + unpack with the calibrated scales for each layer.
  4. Report per-layer reconstruction error (mean abs, p99, max).
  5. Patch each ``k_proj`` / ``v_proj`` to *replace* its output with
     the dequantised version, run the model again, and report whether
     the secret password (``BANANA-7392``) still appears in the output.

This is a correctness probe. If the needle survives a 4K NIAH at
nuq2, the math + scales + Triton kernels all line up and we can
proceed to the harder integration job (paged-KV-cache subclass +
vLLM AttentionImpl). If the needle dies, we know we have a bug to
fix before doing any plumbing.

Usage:
    PYTHONPATH=src python -m kvquant.validate_scales --scales scales/qwen2_5_7b_1m_nuq2_first.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from kvquant.scales import KVScales
from kvquant.triton_kernels import (
    pack_keys_per_channel,
    pack_values_per_token,
    unpack_keys_per_channel,
    unpack_values_per_token,
)


PROMPT_PREAMBLE = "The quick brown fox jumps over the lazy dog. " * 100 + "\n"
NEEDLE = "The secret password is BANANA-7392."
QUESTION = "\n\nQuestion: What is the secret password? Answer in one word: "


def make_niah_prompt(target_tokens: int, tokenizer) -> str:
    chars = target_tokens * 4
    body = (PROMPT_PREAMBLE * (chars // len(PROMPT_PREAMBLE) + 1))[: chars - 200]
    mid = len(body) // 2
    return body[:mid] + "\n\n" + NEEDLE + "\n\n" + body[mid:] + QUESTION


def quantize_dequantize_keys(K_full: torch.Tensor, scale, seqlen: int) -> torch.Tensor:
    """K_full: (T, num_kv_heads, head_dim) fp16. scale: PerChannelScale.

    Returns reconstructed (T, num_kv_heads, head_dim).
    """
    H, D, _ = scale.poles.shape
    out = torch.empty_like(K_full)
    for h in range(H):
        codes, ov, om = pack_keys_per_channel(
            K_full[:, h, :].contiguous(),
            scale.poles[h],
            scale.upper_threshold[h],
            scale.lower_threshold[h],
        )
        out[:, h, :] = unpack_keys_per_channel(codes, ov, om, scale.poles[h])
    return out


def quantize_dequantize_values(V_full: torch.Tensor, scale, seqlen: int) -> torch.Tensor:
    """V_full: (T, num_kv_heads, head_dim). scale: PerTokenScale.

    Per-token scales were calibrated up to ``seqlen_calibration`` (default
    2048). For positions beyond that we re-use the last calibrated
    token's scale (the "tail" hypothesis). Real long-context inference
    needs to re-calibrate or extrapolate cleanly; this validation only
    targets short prompts.
    """
    H, T_cal, _ = scale.poles.shape
    T = V_full.shape[0]
    if T > T_cal:
        # Pad scales by replicating the last token's scale
        pad = T - T_cal
        poles = torch.cat([scale.poles, scale.poles[:, -1:, :].expand(H, pad, -1)], dim=1)
        upper = torch.cat([scale.upper_threshold, scale.upper_threshold[:, -1:].expand(H, pad)], dim=1)
        lower = torch.cat([scale.lower_threshold, scale.lower_threshold[:, -1:].expand(H, pad)], dim=1)
    else:
        poles = scale.poles[:, :T, :]
        upper = scale.upper_threshold[:, :T]
        lower = scale.lower_threshold[:, :T]

    out = torch.empty_like(V_full)
    for h in range(H):
        codes, ov, om = pack_values_per_token(
            V_full[:, h, :].contiguous(),
            poles[h],
            upper[h],
            lower[h],
        )
        out[:, h, :] = unpack_values_per_token(codes, ov, om, poles[h])
    return out


def reconstruction_stats(orig: torch.Tensor, recon: torch.Tensor) -> dict:
    diff = (orig.float() - recon.float()).abs()
    return {
        "mean": diff.mean().item(),
        "p99": diff.flatten().quantile(0.99).item(),
        "max": diff.max().item(),
        "rms": diff.pow(2).mean().sqrt().item(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scales", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    p.add_argument("--target-tokens", type=int, default=2000)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    print(f"loading scales from {args.scales}")
    scales = KVScales.load(args.scales, map_location=args.device)
    scales = scales.to(args.device)
    print(f"  layers: {scales.num_layers}")
    print(f"  num_bits: {scales.num_bits}")

    print(f"loading model {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="eager"
    )
    model.to(args.device)
    model.train(False)

    cfg = model.config
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    prompt = make_niah_prompt(args.target_tokens, tok)
    ids = tok.encode(prompt, return_tensors="pt").to(args.device)
    print(f"prompt: {ids.shape[1]} tokens")

    # --- Pass 1: collect reconstruction stats on the existing forward
    layers = model.model.layers
    stats_K = {i: None for i in range(len(layers))}
    stats_V = {i: None for i in range(len(layers))}

    def make_obs_hook(idx: int, kind: str):
        def hook(_m, _i, output):
            t = output.detach().reshape(output.size(0), output.size(1),
                                         num_kv_heads, head_dim)[0]  # (T, H, D)
            if kind == "K":
                recon = quantize_dequantize_keys(t.contiguous(), scales.per_layer_keys[idx], t.shape[0])
                stats_K[idx] = reconstruction_stats(t, recon)
            else:
                recon = quantize_dequantize_values(t.contiguous(), scales.per_layer_values[idx], t.shape[0])
                stats_V[idx] = reconstruction_stats(t, recon)
        return hook

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_obs_hook(i, "K")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_obs_hook(i, "V")))

    print("\n[1/2] observing reconstruction errors")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=16, do_sample=False)
    print(f"  generation took {time.time()-t0:.1f}s")
    completion = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"  baseline completion: {completion[:80]!r}")
    baseline_needle = "BANANA" in completion.upper()
    print(f"  baseline needle: {'YES' if baseline_needle else 'NO'}")
    for h in handles:
        h.remove()

    # Layer-by-layer summary
    print("\nKey reconstruction:")
    print(f"  {'layer':>5} {'mean':>9} {'p99':>9} {'max':>9} {'rms':>9}")
    for i in range(len(layers)):
        s = stats_K[i]
        print(f"  {i:>5} {s['mean']:>9.4f} {s['p99']:>9.4f} {s['max']:>9.4f} {s['rms']:>9.4f}")
    print("\nValue reconstruction:")
    print(f"  {'layer':>5} {'mean':>9} {'p99':>9} {'max':>9} {'rms':>9}")
    for i in range(len(layers)):
        s = stats_V[i]
        print(f"  {i:>5} {s['mean']:>9.4f} {s['p99']:>9.4f} {s['max']:>9.4f} {s['rms']:>9.4f}")

    # --- Pass 2: actually swap KV in the forward path, see if needle survives.
    print("\n[2/2] running with quantised KV in the forward path")

    def make_replace_hook(idx: int, kind: str):
        def hook(_m, _i, output):
            B, T = output.size(0), output.size(1)
            t = output.reshape(B, T, num_kv_heads, head_dim)[0]
            if kind == "K":
                recon = quantize_dequantize_keys(t.contiguous(), scales.per_layer_keys[idx], T)
            else:
                recon = quantize_dequantize_values(t.contiguous(), scales.per_layer_values[idx], T)
            return recon.unsqueeze(0).reshape(B, T, num_kv_heads * head_dim)
        return hook

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_replace_hook(i, "K")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_replace_hook(i, "V")))

    t0 = time.time()
    with torch.no_grad():
        out_q = model.generate(ids, max_new_tokens=16, do_sample=False)
    print(f"  generation took {time.time()-t0:.1f}s")
    completion_q = tok.decode(out_q[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"  quantised completion: {completion_q[:80]!r}")
    q_needle = "BANANA" in completion_q.upper()
    print(f"  quantised needle: {'YES' if q_needle else 'NO'}")

    print("\n" + "=" * 50)
    print(f" baseline needle: {'YES' if baseline_needle else 'NO'}")
    print(f" nuq2 needle:     {'YES' if q_needle else 'NO'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
