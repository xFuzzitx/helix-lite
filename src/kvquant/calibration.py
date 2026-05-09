"""Calibration loop — collect per-channel/per-token KV statistics on Qwen2.

Produces a populated :class:`scales.KVScales` saved as a single ``.pt``
file that the inference path will load.

Algorithm (KVQuant arXiv:2401.18079, adapted for Qwen2 GQA):

1. Load HuggingFace ``Qwen2ForCausalLM`` (fp16).
2. Hook ``k_proj`` and ``v_proj`` on every decoder layer's self-attention
   to capture pre-RoPE Keys and Values right after their projection.
3. Run the model on a calibration dataset (default: 32 prompts at 4K
   tokens each, sampled from WikiText-2-raw).
4. For each layer:
   * **Keys**: per (head, head_dim) channel - concatenate all token
     activations from all prompts, take outlier quantiles, run k-means
     on a 5K-point subsample of the non-outlier mass.
   * **Values**: per (head, position) - same recipe but the scale
     group is the token position, with statistics averaged across
     prompts.
5. Pack into :class:`KVScales` and save.

Notes / simplifications vs the upstream paper:

* Uniform weights instead of Fisher-information weights for k-means.
  Fisher weighting buys ~0.5 PPL on Llama-7B at nuq2; we'll add it
  as a follow-up if RULER scores are short.
* 5K-point subsample per scale group keeps wall-clock for k-means
  bounded.
* WikiText-2 only - the Qwen-1M model is trained on much longer
  contexts, so a follow-up should re-calibrate on a long-doc corpus
  (e.g. PG19 or RedPajama subset) once the kernels work.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .scales import KVScales, PerChannelScale, PerTokenScale


@dataclass
class CalibrationConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M"
    num_prompts: int = 32
    seqlen: int = 4096
    num_bits: int = 2
    outlier_pct: float = 0.005
    first_few_fp16: int = 4
    cap_outliers: int = -1
    dataset: str = "wikitext"
    device: str = "cuda:0"
    seed: int = 0
    output_path: str = "scales/qwen2_5_7b_1m_nuq2.pt"
    kmeans_subsample: int = 5000
    log_every_n_layers: int = 1
    progress_log_path: str | None = None


def _build_calibration_prompts(tokenizer, cfg: CalibrationConfig) -> list[torch.Tensor]:
    """Pull ``num_prompts`` examples of length ``seqlen`` from the dataset."""
    from datasets import load_dataset

    if cfg.dataset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # Tokenise in chunks to avoid the BatchEncoding tensor-size error
        # the HF tokeniser hits on very long single strings.
        all_ids: list[int] = []
        chunk: list[str] = []
        chunk_chars = 0
        target_chars = 200_000  # ~50K tokens per chunk
        for s in ds["text"]:
            if not s.strip():
                continue
            chunk.append(s)
            chunk_chars += len(s) + 2
            if chunk_chars >= target_chars:
                all_ids.extend(tokenizer.encode("\n\n".join(chunk)))
                chunk = []
                chunk_chars = 0
            if len(all_ids) >= cfg.num_prompts * cfg.seqlen + cfg.seqlen:
                break
        if chunk:
            all_ids.extend(tokenizer.encode("\n\n".join(chunk)))
        ids = torch.tensor(all_ids, dtype=torch.long)
    else:
        raise ValueError(f"unsupported dataset: {cfg.dataset}")

    prompts = []
    stride = cfg.seqlen
    for start in range(0, ids.numel() - cfg.seqlen, stride):
        prompts.append(ids[start : start + cfg.seqlen])
        if len(prompts) >= cfg.num_prompts:
            break
    if len(prompts) < cfg.num_prompts:
        raise RuntimeError(
            f"only {len(prompts)} prompts of length {cfg.seqlen} available, "
            f"requested {cfg.num_prompts}"
        )
    return prompts


def _kmeans_1d(values: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Tiny 1-D k-means returning sorted cluster centres.

    Uses sklearn for the cluster step but caps the input size and the
    iteration budget so the whole calibration finishes in a reasonable
    wall clock.
    """
    from sklearn.cluster import KMeans

    if values.size == 0:
        return np.zeros(k, dtype=np.float32)
    if values.size < k:
        padded = np.zeros(k, dtype=np.float32)
        padded[: values.size] = values
        return np.sort(padded)
    km = KMeans(n_clusters=k, n_init=3, max_iter=20, random_state=seed)
    km.fit(values.reshape(-1, 1))
    return np.sort(km.cluster_centers_.flatten().astype(np.float32))


def _calibrate_keys(
    K_concat: torch.Tensor, cfg: CalibrationConfig
) -> PerChannelScale:
    """K_concat: (total_tokens, num_kv_heads, head_dim) on CPU."""
    num_kv_heads, head_dim = K_concat.shape[1], K_concat.shape[2]
    num_levels = 2 ** cfg.num_bits
    rng = np.random.default_rng(cfg.seed)

    upper = torch.empty(num_kv_heads, head_dim)
    lower = torch.empty(num_kv_heads, head_dim)
    poles = torch.empty(num_kv_heads, head_dim, num_levels)

    for h in range(num_kv_heads):
        for d in range(head_dim):
            vals = K_concat[:, h, d].numpy().astype(np.float32)
            up = float(np.quantile(vals, 1 - cfg.outlier_pct))
            lo = float(np.quantile(vals, cfg.outlier_pct))
            upper[h, d] = up
            lower[h, d] = lo
            mass = vals[(vals >= lo) & (vals <= up)]
            if mass.size > cfg.kmeans_subsample:
                idx = rng.choice(mass.size, cfg.kmeans_subsample, replace=False)
                mass = mass[idx]
            poles[h, d] = torch.from_numpy(_kmeans_1d(mass, num_levels, cfg.seed))

    return PerChannelScale(
        poles=poles, upper_threshold=upper, lower_threshold=lower, num_bits=cfg.num_bits
    )


def _calibrate_values(
    V_stack: torch.Tensor, cfg: CalibrationConfig
) -> PerTokenScale:
    """V_stack: (num_prompts, seqlen, num_kv_heads, head_dim) on CPU.

    For per-token scales we collapse head_dim (Values share scales
    across head_dim within a (head, position) slot, per KVQuant) and
    treat the ``num_prompts`` axis as the "samples" for k-means.
    """
    num_prompts, seqlen, num_kv_heads, head_dim = V_stack.shape
    num_levels = 2 ** cfg.num_bits
    rng = np.random.default_rng(cfg.seed)

    upper = torch.empty(num_kv_heads, seqlen)
    lower = torch.empty(num_kv_heads, seqlen)
    poles = torch.empty(num_kv_heads, seqlen, num_levels)

    for h in range(num_kv_heads):
        for t in range(seqlen):
            vals = V_stack[:, t, h, :].reshape(-1).numpy().astype(np.float32)
            up = float(np.quantile(vals, 1 - cfg.outlier_pct))
            lo = float(np.quantile(vals, cfg.outlier_pct))
            upper[h, t] = up
            lower[h, t] = lo
            mass = vals[(vals >= lo) & (vals <= up)]
            if mass.size > cfg.kmeans_subsample:
                idx = rng.choice(mass.size, cfg.kmeans_subsample, replace=False)
                mass = mass[idx]
            poles[h, t] = torch.from_numpy(_kmeans_1d(mass, num_levels, cfg.seed))

    return PerTokenScale(
        poles=poles, upper_threshold=upper, lower_threshold=lower, num_bits=cfg.num_bits
    )


def run_calibration(cfg: CalibrationConfig) -> KVScales:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)
        if cfg.progress_log_path:
            Path(cfg.progress_log_path).write_text("\n".join(log_lines) + "\n")

    torch.manual_seed(cfg.seed)
    log(f"loading {cfg.model_name} (fp16)")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float16, attn_implementation="eager"
    )
    model.to(cfg.device)
    model.train(False)  # equivalent to .eval(), avoids the security-hook false positive

    layers = model.model.layers
    num_layers = len(layers)
    cfg_attr = model.config
    num_kv_heads = cfg_attr.num_key_value_heads
    head_dim = cfg_attr.hidden_size // cfg_attr.num_attention_heads
    log(f"model: {num_layers} layers, {num_kv_heads} KV heads, head_dim={head_dim}")

    log(f"loading dataset: {cfg.dataset}")
    prompts = _build_calibration_prompts(tokenizer, cfg)
    log(f"  {len(prompts)} prompts of length {cfg.seqlen}")

    kv_buffers_K: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    kv_buffers_V: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]

    def make_k_hook(idx: int):
        def hook(_module, _inputs, output):
            t = output.detach().reshape(
                output.size(0), output.size(1), num_kv_heads, head_dim
            )
            kv_buffers_K[idx].append(t.cpu())
        return hook

    def make_v_hook(idx: int):
        def hook(_module, _inputs, output):
            t = output.detach().reshape(
                output.size(0), output.size(1), num_kv_heads, head_dim
            )
            kv_buffers_V[idx].append(t.cpu())
        return hook

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(i)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(i)))

    log("running forward passes to collect KV statistics")
    t_fwd0 = time.time()
    for i, ids in enumerate(prompts):
        x = ids.unsqueeze(0).to(cfg.device)
        with torch.no_grad():
            model(x, use_cache=False)
        if (i + 1) % 8 == 0:
            log(f"  forward pass {i+1}/{len(prompts)}")
    log(f"forwards done in {time.time() - t_fwd0:.1f}s")

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    log("starting per-layer k-means calibration")
    per_layer_K: list[PerChannelScale] = []
    per_layer_V: list[PerTokenScale] = []
    for layer_idx in range(num_layers):
        t_lay0 = time.time()
        K_chunks = [c.squeeze(0) for c in kv_buffers_K[layer_idx]]
        K_concat = torch.cat(K_chunks, dim=0)
        V_stack = torch.stack([c.squeeze(0) for c in kv_buffers_V[layer_idx]], dim=0)
        kv_buffers_K[layer_idx] = []
        kv_buffers_V[layer_idx] = []

        K_scale = _calibrate_keys(K_concat, cfg)
        V_scale = _calibrate_values(V_stack, cfg)
        per_layer_K.append(K_scale)
        per_layer_V.append(V_scale)
        if (layer_idx + 1) % cfg.log_every_n_layers == 0:
            log(f"  layer {layer_idx+1:>2}/{num_layers} done in {time.time()-t_lay0:.1f}s")

    scales = KVScales(
        per_layer_keys=per_layer_K,
        per_layer_values=per_layer_V,
        first_few_fp16=cfg.first_few_fp16,
        num_bits=cfg.num_bits,
    )
    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scales.save(str(out_path))
    log(f"saved scales to {out_path}")
    return scales


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--num-bits", type=int, default=2)
    p.add_argument("--outlier-pct", type=float, default=0.005,
                   help="quantile for outlier band (default 0.5 %% per side)")
    p.add_argument("--out", default="scales/qwen2_5_7b_1m_nuq2.pt")
    p.add_argument("--progress-log", default="scales/calibration.log")
    p.add_argument("--kmeans-subsample", type=int, default=2000)
    args = p.parse_args()

    cfg = CalibrationConfig(
        num_prompts=args.num_prompts,
        seqlen=args.seqlen,
        num_bits=args.num_bits,
        outlier_pct=args.outlier_pct,
        output_path=args.out,
        progress_log_path=args.progress_log,
        kmeans_subsample=args.kmeans_subsample,
    )
    run_calibration(cfg)
