"""Stitch two KVScales files into a mixed-bit calibration.

Use case: nuq2 holds quality for shallow layers but breaks at deep
layers (V mean error blows up at L20+ on Qwen2.5-7B). Mixed-bit
keeps the cheap nuq2 layers and swaps in nuq4 only where it's
required, holding most of the compression with the quality recovery
on the layers that actually need it.

The runtime ``KVQuantAttentionImpl`` already supports per-layer scale
shapes — each layer's `poles` carries its own `NUM_LEVELS` along the
last dim, and the Triton kernels read that as a `constexpr` per
launch. The only thing we have to be honest about is the top-level
``num_bits`` summary on the ``KVScales`` dataclass; we set it to the
SHALLOW value so the runtime print is informative ("nuq2 mixed").

Usage::

    python -m benchmarks.pr1c.mix_scales \
        --shallow scales/qwen2_5_7b_1m_nuq2_v3.pt \
        --deep    scales/qwen2_5_7b_1m_nuq4_v3.pt \
        --cut 20 \
        --out scales/qwen2_5_7b_1m_mixed_n2_n4_cut20.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kvquant.scales import KVScales


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shallow", required=True, type=Path,
                   help="lower-bit scales used for layers [0, cut)")
    p.add_argument("--deep", required=True, type=Path,
                   help="higher-bit scales used for layers [cut, num_layers)")
    p.add_argument("--cut", required=True, type=int,
                   help="first layer index that uses the deep (higher-bit) scales")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    shallow = KVScales.load(str(args.shallow))
    deep = KVScales.load(str(args.deep))
    assert shallow.num_layers == deep.num_layers, (
        f"layer count mismatch: shallow={shallow.num_layers} deep={deep.num_layers}"
    )
    L = shallow.num_layers
    assert 0 < args.cut < L, f"--cut must be in (0, {L}); got {args.cut}"

    print(f"shallow ({shallow.num_bits}-bit) for layers [0, {args.cut})")
    print(f"deep    ({deep.num_bits}-bit) for layers [{args.cut}, {L})")

    mixed = KVScales(
        per_layer_keys=(
            shallow.per_layer_keys[: args.cut] + deep.per_layer_keys[args.cut:]
        ),
        per_layer_values=(
            shallow.per_layer_values[: args.cut] + deep.per_layer_values[args.cut:]
        ),
        first_few_fp16=shallow.first_few_fp16,
        num_bits=shallow.num_bits,  # nominal; actual is per-layer in poles.shape[-1]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mixed.save(str(args.out))

    print(f"\nsaved {args.out}")
    print("per-layer levels:")
    for i, k in enumerate(mixed.per_layer_keys):
        v = mixed.per_layer_values[i]
        mark = "•" if i < args.cut else "*"
        print(f"  {mark} L{i:>2}  K levels={k.poles.shape[-1]:>2}  "
              f"V levels={v.poles.shape[-1]:>2}")


if __name__ == "__main__":
    main()
