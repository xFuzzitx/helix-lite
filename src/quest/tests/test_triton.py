"""Triton parity test for the Quest upper-bound kernel.

The Triton kernel must match the pure-PyTorch reference within fp16
slack. Run on whatever GPU is currently visible.

Run:
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python src/quest/tests/test_triton.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest.reference import page_upper_bound
from quest.triton_kernels import page_upper_bound_triton


def test_matches_reference_qwen_shape() -> None:
    """Match the actual Qwen2.5-7B-1M decode shapes: 4 KV heads, head_dim 128,
    a few hundred pages."""
    torch.manual_seed(0)
    P, H, D = 256, 4, 128
    q = torch.randn(H, D, dtype=torch.float16, device="cuda")
    K_min = torch.randn(P, H, D, dtype=torch.float16, device="cuda")
    K_max = K_min + torch.rand(P, H, D, dtype=torch.float16, device="cuda")

    out_ref = page_upper_bound(q, K_min, K_max)
    out_tri = page_upper_bound_triton(q, K_min, K_max)
    diff = (out_ref - out_tri).abs()
    max_err = diff.max().item()
    # Relative error only meaningful where the magnitude is large enough
    # to dominate fp16 quantum (~5e-2 for values around 1). Filter near-zero.
    significant = out_ref.abs() > 1.0
    rel_err = 0.0
    if significant.any():
        rel_err = (diff[significant] / out_ref.abs()[significant]).max().item()
    assert max_err < 0.5, f"max abs diff {max_err:.4f} exceeds tolerance"
    assert rel_err < 0.05, f"max rel diff (on significant entries) {rel_err:.4e}"
    # Also: top-K agreement is what actually matters for Quest. Verify
    # the top-32 page indices are the same across paths (allow 2 swaps).
    top_ref = out_ref.topk(k=32, dim=0).indices
    top_tri = out_tri.topk(k=32, dim=0).indices
    set_ref = [set(top_ref[:, h].tolist()) for h in range(H)]
    set_tri = [set(top_tri[:, h].tolist()) for h in range(H)]
    for h in range(H):
        sym_diff = len(set_ref[h] ^ set_tri[h])
        assert sym_diff <= 4, f"head {h}: top-32 disagreement {sym_diff}"


def test_matches_reference_small() -> None:
    torch.manual_seed(1)
    P, H, D = 16, 2, 32
    q = torch.randn(H, D, dtype=torch.float16, device="cuda")
    K_min = torch.randn(P, H, D, dtype=torch.float16, device="cuda")
    K_max = K_min + torch.rand(P, H, D, dtype=torch.float16, device="cuda")
    out_ref = page_upper_bound(q, K_min, K_max)
    out_tri = page_upper_bound_triton(q, K_min, K_max)
    assert torch.allclose(out_ref, out_tri, atol=0.1, rtol=0.01)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        sys.exit(0)
    tests = [
        ("matches_reference_qwen_shape", test_matches_reference_qwen_shape),
        ("matches_reference_small", test_matches_reference_small),
    ]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok  {name}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL {name}: {e}")
        except Exception as e:
            fails += 1
            print(f"  ERR  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
