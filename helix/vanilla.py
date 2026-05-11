"""Vanilla AWQ-INT4 query path: load via vLLM, answer with the full doc.

Uses the same engine config we benchmarked at PR1a (8/8 multi-needle
recall at 32K and 128K, single 3090, GPU 1 free). The function below
is intentionally minimal: load once, call :func:`vanilla.query` per
question.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class VanillaConfig:
    model: str = "graelo/Qwen2.5-7B-Instruct-1M-AWQ"
    max_model_len: int = 128_000
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    # If set, install the KVQuant math-wrapper before LLM() so every K/V
    # round-trips through the calibrated nuq quant→dequant. Cache stays
    # fp16 (no VRAM win yet — that's Phase 1B). Use to validate calibration
    # quality at scale, or to A/B against the baseline AWQ path.
    nuq_scales: str | None = None
    nuq_device: str = "cuda:0"


class VanillaSession:
    """Loaded model + persistent state, kept across multiple queries."""

    def __init__(self, cfg: VanillaConfig | None = None) -> None:
        self.cfg = cfg or VanillaConfig()
        if self.cfg.tensor_parallel_size > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in range(self.cfg.tensor_parallel_size)
            )
        else:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        if self.cfg.nuq_scales:
            # Must run before LLM() constructs the attention backend.
            import sys
            from pathlib import Path
            src = Path(__file__).resolve().parent.parent / "src"
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            from kvquant.vllm_backend import install_kvquant_backend
            install_kvquant_backend(self.cfg.nuq_scales, device=self.cfg.nuq_device)

        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model)
        self.llm = LLM(
            model=self.cfg.model,
            dtype="auto",
            max_model_len=self.cfg.max_model_len,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            enforce_eager=self.cfg.enforce_eager,
            trust_remote_code=False,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
        )

    def query(self, question: str, document: str | None = None,
              max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Answer ``question`` optionally grounded in ``document``.

        The prompt format is the simplest one that works on Qwen2.5
        instruct: a single user turn that pastes the doc above the
        question.
        """
        from vllm import SamplingParams

        if document:
            prompt = f"{document}\n\n---\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"

        sampling = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, top_p=1.0,
            stop=["\n\nQuestion:", "\n\n---"],
        )
        out = self.llm.generate([prompt], sampling)
        return out[0].outputs[0].text.strip()


def vanilla(question: str, document: str | None = None,
            cfg: VanillaConfig | None = None, **gen_kwargs: object) -> str:
    """One-shot helper: load, query, drop. Use :class:`VanillaSession`
    if you plan to ask multiple questions back to back."""
    session = VanillaSession(cfg)
    return session.query(question, document, **gen_kwargs)
