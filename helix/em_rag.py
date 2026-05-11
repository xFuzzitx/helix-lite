"""EM-LLM-style retrieval-augmented querying.

Document is segmented via Bayesian surprise on the per-token logits,
each segment becomes an Episode with a max-abs-pooled embedding on
GPU 1, the question embedding is computed the same way, and the top-M
episodes are concatenated above the question for the AWQ model to
generate from.

This is the *text-level* version of EM-LLM: we stuff the retrieved
text back into the model's context. The KV-level swap (assemble_kv
in src/emllm/hot_swap.py) gives the same answers but in O(hot +
M*episode_len) per decode step instead of O(retrieved_text); that
optimisation is the deferred vLLM AttentionImpl integration.

Two-pass execution:

  Pass 1 (HF transformers, bf16, sdpa):
    forward over the document with output_hidden_states=True,
    capture last-layer hidden + per-step logits, segment, build the
    store. Free the model.

  Pass 2 (vLLM AWQ on the same or a separate GPU):
    build prompt = [retrieved episodes' text ; question], generate.

This uses GPU 0 + GPU 1 in sequence: HF Qwen2 on GPU 0 for indexing
(and embeddings on GPU 1), then HF released, then vLLM AWQ on GPU 0
for generation. If you want both resident at once, set
``cfg.indexer_device='cuda:1'`` so HF lives on GPU 1.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

# Ensure src/ is importable so emllm/ resolves
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from emllm.episode_store import Episode  # noqa: E402
from emllm.kv_store import KVEpisodeStore  # noqa: E402
from emllm.segmenter import BayesianSurpriseSegmenter, SegmenterConfig  # noqa: E402


@dataclass
class EMRAGConfig:
    indexer_model: str = "Qwen/Qwen2.5-7B-Instruct-1M"
    generator_model: str = "graelo/Qwen2.5-7B-Instruct-1M-AWQ"
    indexer_device: str = "cuda:0"
    store_device: str = "cuda:1"
    indexer_dtype: torch.dtype = torch.bfloat16
    threshold_quantile: float = 0.95
    min_seg: int = 128
    max_seg: int = 512
    top_m: int = 16
    pool: str = "max-abs"           # "max-abs" | "mean" | "last" | "mean+max"
    query_pool: str = "last"        # "max-abs" | "mean" | "last" | "mean+max"
    pool_alpha: float = 0.5         # weight for max-abs term in "mean+max"
    # Layer used for episode/query embeddings. Empirically (top-m=64 multi-needle NIAH,
    # Qwen2.5-7B-1M), "mid" (~layer 14/28) gives 2× recall @128K vs "last" because the
    # last layer is specialised for next-token prediction; mid-block hidden states carry
    # the semantic content needed for retrieval. See README "Eval results".
    indexer_layer: str = "mid"      # "last" | "mid" | "multi-last4" | "multi-mid4"
    chunk_size: int = 8192          # how many tokens of doc to forward at once
    max_doc_tokens: int = 200_000   # safety cap on indexed length
    # HyDE (Hypothetical Document Embeddings) query expansion: generate a short
    # hypothetical answer to the question with the indexer model and embed the
    # full (template + question + answer) string. The hypothetical text bridges
    # the lexical/style gap between the question form and the doc-passage form
    # of the needle. Costs ~1 forward.generate(max_new_tokens=hyde_max_tokens)
    # per query.
    hyde: bool = False
    hyde_max_tokens: int = 48
    hyde_template: str = (
        "Below is a question and a likely passage that would contain its answer.\n\n"
        "Question: {question}\nPassage: "
    )
    # Cross-encoder / LLM-as-reranker: take a wider cosine top-K, then re-score
    # each candidate by asking the indexer model whether the question's answer
    # is in the candidate text. Final top-M = the highest-scored after rerank.
    # Score = logit(" yes") − logit(" no") at the answer position.
    rerank: bool = False
    rerank_topk: int = 256              # cosine top-K BEFORE rerank
    rerank_max_episode_chars: int = 2000  # truncate long episodes for the rerank prompt
    rerank_template: str = (
        "Read the document below and judge whether it contains the answer to the question.\n\n"
        "Document: {episode}\n\nQuestion: {question}\n\nAnswer (yes/no):"
    )
    # Dedicated similarity encoder for episode/query embeddings. Bypasses
    # the indexer hidden-state pooling entirely. Examples:
    #   "BAAI/bge-m3"           (multilingual, 568M, 1024-dim, 8K input)
    #   "Qwen/Qwen3-Embedding-0.6B" (Qwen-aligned, 600M, 1024-dim, 32K input)
    # Lives on embedding_device (GPU 1 by default, so it co-resides with the
    # store and frees GPU 0 for the indexer/AWQ).
    embedding_model: str | None = None
    embedding_device: str = "cuda:1"
    embedding_max_tokens: int = 8192
    embedding_batch_size: int = 32


@dataclass
class EpisodeIndex:
    """Persistent index built from one document.

    ``token_ids`` keeps the full token sequence so we can decode
    retrieved spans back to text at query time.
    """

    token_ids: list[int]
    store: KVEpisodeStore
    boundaries: list[int]


def _pool_hidden(slab: torch.Tensor, mode: str, alpha: float = 0.5) -> torch.Tensor:
    if mode == "mean":
        return slab.mean(dim=0)
    if mode == "last":
        return slab[-1]
    if mode == "max-abs":
        idx = slab.abs().argmax(dim=0)
        return slab.gather(0, idx.unsqueeze(0)).squeeze(0)
    if mode == "mean+max":
        m = slab.mean(dim=0)
        idx = slab.abs().argmax(dim=0)
        x = slab.gather(0, idx.unsqueeze(0)).squeeze(0)
        return m + alpha * x
    raise ValueError(f"unknown pool mode: {mode}")


def _select_hidden_chunk(hidden_states: tuple, mode: str) -> torch.Tensor:
    """Pick the hidden tensor (T, H) from a forward's hidden_states tuple.

    ``hidden_states`` is ``(num_layers + 1)`` tensors of shape ``(1, T, H)``;
    index 0 is the embedding output, indices 1..N are layer outputs.
    """
    if mode == "last":
        return hidden_states[-1][0]
    if mode == "mid":
        return hidden_states[len(hidden_states) // 2][0]
    if mode == "multi-last4":
        stacked = torch.stack([hidden_states[-i][0] for i in (1, 2, 3, 4)], dim=0)
        return stacked.mean(dim=0)
    if mode == "multi-mid4":
        mid = len(hidden_states) // 2
        stacked = torch.stack(
            [hidden_states[mid - 1][0], hidden_states[mid][0],
             hidden_states[mid + 1][0], hidden_states[mid + 2][0]],
            dim=0,
        )
        return stacked.mean(dim=0)
    raise ValueError(f"unknown indexer_layer mode: {mode}")


def _load_indexer(cfg: EMRAGConfig):
    """Load the HF indexer model+tokenizer onto ``cfg.indexer_device``.

    Returns ``(model, tokenizer)``. Caller is responsible for freeing
    the model (``del`` + ``torch.cuda.empty_cache()``).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.indexer_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.indexer_model, dtype=cfg.indexer_dtype, attn_implementation="sdpa"
    )
    model.to(cfg.indexer_device)
    model.train(False)
    return model, tok


def _load_embedding_model(cfg: EMRAGConfig):
    """Load a dedicated similarity encoder onto ``cfg.embedding_device``.

    Returns ``(model, tokenizer)`` or ``(None, None)`` if no embedding
    model is configured. The encoder is meant to replace the indexer
    hidden-state pooling for episode/query embeddings.
    """
    if not cfg.embedding_model:
        return None, None
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.embedding_model)
    model = AutoModel.from_pretrained(
        cfg.embedding_model, dtype=cfg.indexer_dtype
    )
    model.to(cfg.embedding_device)
    model.train(False)
    return model, tok


def _encode_texts(
    model, tokenizer, texts: list[str], cfg: EMRAGConfig,
) -> torch.Tensor:
    """Batched encoder forward → ``(N, emb_dim)`` L2-normalized embeddings.

    Uses the CLS-token (BGE-style) pooling: first-position last-layer
    hidden state. Works for BGE-M3, Qwen3-Embedding, and most sentence
    encoder checkpoints; falls back gracefully if there's no CLS-style
    pooling (the model still returns *some* sequence-level vector).
    """
    embs: list[torch.Tensor] = []
    bs = cfg.embedding_batch_size
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=cfg.embedding_max_tokens,
            return_tensors="pt",
        ).to(cfg.embedding_device)
        with torch.no_grad():
            out = model(**inputs)
        # Take CLS / first-token embedding from last hidden state.
        e = out.last_hidden_state[:, 0, :]
        e = torch.nn.functional.normalize(e.float(), p=2, dim=-1)
        embs.append(e.to(cfg.indexer_dtype))
    return torch.cat(embs, dim=0)


def build_index(document: str, cfg: EMRAGConfig | None = None,
                *, model=None, tokenizer=None,
                embedding_model=None, embedding_tokenizer=None) -> EpisodeIndex:
    """Run the indexing forward and return a populated store.

    If ``model``/``tokenizer`` are passed, reuse them and do NOT free
    them at the end (caller-owned). Otherwise, load+free internally.
    Same convention for ``embedding_model``/``embedding_tokenizer``
    when ``cfg.embedding_model`` is set.
    """
    cfg = cfg or EMRAGConfig()
    owns_model = model is None
    if owns_model:
        model, tokenizer = _load_indexer(cfg)
    tok = tokenizer

    ids = tok.encode(document)[: cfg.max_doc_tokens]
    if not ids:
        raise ValueError("empty document")
    ids_tensor = torch.tensor(ids, dtype=torch.long, device=cfg.indexer_device).unsqueeze(0)
    T = ids_tensor.shape[1]

    seg_cfg = SegmenterConfig(
        threshold_quantile=cfg.threshold_quantile,
        window=128,
        min_segment_len=cfg.min_seg,
        max_segment_len=cfg.max_seg,
    )
    segmenter = BayesianSurpriseSegmenter(seg_cfg)

    hidden_size = model.config.hidden_size
    store = KVEpisodeStore(
        emb_dim=hidden_size, device=cfg.store_device,
        capacity=max(1024, T // cfg.min_seg + 16),
        dtype=cfg.indexer_dtype,
    )

    # Forward in chunks so we don't OOM on huge documents. We don't
    # use HF's KV cache here; segmentation only needs logits and a
    # last-layer hidden state per token.
    boundaries: list[int] = []
    seg_start = 0
    last_hidden_chunks: list[torch.Tensor] = []
    pos = 0
    while pos < T:
        end = min(pos + cfg.chunk_size, T)
        chunk = ids_tensor[:, pos:end]
        with torch.no_grad():
            out = model(chunk, output_hidden_states=True, use_cache=False)
        chunk_logits = out.logits[0]
        chunk_hidden = _select_hidden_chunk(out.hidden_states, cfg.indexer_layer)
        # Walk per-token; close any segments that boundary out
        for i in range(chunk_logits.shape[0]):
            t = pos + i
            b = segmenter.step(chunk_logits[i])
            if b is not None:
                boundaries.append(b.position)
                # Build the episode from the spanned hidden states
                global_slab = _gather_hidden_for_range(
                    last_hidden_chunks + [chunk_hidden],
                    seg_start, b.position, base_offset=pos,
                )
                emb = _pool_hidden(global_slab, cfg.pool, cfg.pool_alpha)
                ep = store.add(emb, (seg_start, b.position))
                # We do NOT store KV here - the text-level RAG pass
                # uses the AWQ generator separately. KV-level swap
                # is the deferred vLLM AttentionImpl integration.
                seg_start = b.position
        last_hidden_chunks.append(chunk_hidden)
        # Bound memory: keep only enough hidden for the open segment
        retained_start = seg_start - pos
        if retained_start > 0:
            # We can drop everything before the open segment; keep only
            # the slice covering [seg_start, end).
            last_hidden_chunks = [chunk_hidden[max(0, retained_start):]]
        pos = end

    # Close the final open segment
    if seg_start < T:
        global_slab = _gather_hidden_for_range(
            last_hidden_chunks, seg_start, T, base_offset=T - last_hidden_chunks[-1].shape[0]
        )
        emb = _pool_hidden(global_slab, cfg.pool)
        store.add(emb, (seg_start, T))
        boundaries.append(T)

    if owns_model:
        del model
        torch.cuda.empty_cache()

    if cfg.embedding_model:
        store = _rebuild_store_with_embedding_model(
            store, ids, tok, cfg,
            embedding_model=embedding_model,
            embedding_tokenizer=embedding_tokenizer,
        )
    return EpisodeIndex(token_ids=ids, store=store, boundaries=boundaries)


def _rebuild_store_with_embedding_model(
    store: "KVEpisodeStore", token_ids: list[int],
    indexer_tokenizer, cfg: EMRAGConfig,
    *, embedding_model=None, embedding_tokenizer=None,
) -> "KVEpisodeStore":
    """Re-embed every episode in ``store`` using the dedicated encoder.

    Returns a fresh store whose embeddings are produced by
    ``cfg.embedding_model``. Episode token_ranges and surprise values
    are preserved. Reuses caller-owned ``embedding_model``/``tokenizer``
    when provided; otherwise loads+frees internally.
    """
    owns_emb = embedding_model is None
    if owns_emb:
        emb_model, emb_tok = _load_embedding_model(cfg)
    else:
        emb_model, emb_tok = embedding_model, embedding_tokenizer
    try:
        texts = [
            indexer_tokenizer.decode(token_ids[ep.token_range[0] : ep.token_range[1]])
            for ep in store.episodes
        ]
        embs = _encode_texts(emb_model, emb_tok, texts, cfg)
    finally:
        if owns_emb:
            del emb_model
            torch.cuda.empty_cache()

    emb_dim = embs.shape[-1]
    new_store = KVEpisodeStore(
        emb_dim=emb_dim, device=cfg.store_device,
        capacity=store.capacity, dtype=cfg.indexer_dtype,
    )
    for i, ep in enumerate(store.episodes):
        new_store.add(embs[i], ep.token_range, ep.surprise)
    return new_store


def _gather_hidden_for_range(chunks: list[torch.Tensor], start: int, end: int,
                              base_offset: int) -> torch.Tensor:
    """Slice ``chunks`` to recover hidden states for absolute token range.

    Each chunk is a fp16/bf16 tensor of shape ``(T_chunk, hidden)``;
    they are stored in order. ``base_offset`` is the absolute token
    position of the *first* chunk's first row. We assume the requested
    range is contained within the available chunks; in practice the
    segmenter only ever closes segments whose tokens we've seen.
    """
    parts: list[torch.Tensor] = []
    pos = base_offset
    for c in chunks:
        c_end = pos + c.shape[0]
        if c_end <= start:
            pos = c_end
            continue
        if pos >= end:
            break
        a = max(0, start - pos)
        b = min(c.shape[0], end - pos)
        parts.append(c[a:b])
        pos = c_end
    if not parts:
        # Edge case: the whole range was thrown away; return a zero vector
        return torch.zeros((1, chunks[-1].shape[1]),
                           device=chunks[-1].device, dtype=chunks[-1].dtype)
    return torch.cat(parts, dim=0)


def retrieve_episode_texts(
    index: EpisodeIndex, question: str, cfg: EMRAGConfig | None = None,
    *, model=None, tokenizer=None,
    embedding_model=None, embedding_tokenizer=None,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Embed ``question`` with the same indexer and return the top-M
    episodes' decoded text (sorted by token range for causal order).

    Pass ``model``/``tokenizer`` to reuse a pre-loaded indexer (avoids
    a second 14 GB load). Otherwise loads+frees internally.
    Same convention for ``embedding_model``/``embedding_tokenizer``
    when ``cfg.embedding_model`` is set.
    """
    cfg = cfg or EMRAGConfig()
    owns_model = model is None
    if owns_model:
        model, tokenizer = _load_indexer(cfg)

    # Dedicated embedding model path: skip Qwen indexer forward entirely
    # for the query and encode it with the same encoder used at index time.
    if cfg.embedding_model:
        owns_emb = embedding_model is None
        if owns_emb:
            emb_model, emb_tok = _load_embedding_model(cfg)
        else:
            emb_model, emb_tok = embedding_model, embedding_tokenizer
        try:
            q_emb = _encode_texts(emb_model, emb_tok, [question], cfg)[0]
        finally:
            if owns_emb:
                del emb_model
                torch.cuda.empty_cache()
        cosine_k = cfg.rerank_topk if cfg.rerank else cfg.top_m
        top = index.store.topk(q_emb, k=cosine_k, metric="cosine")
        eps: list[Episode] = [ep for ep, _ in top]
        if cfg.rerank:
            scored = _rerank_with_model(
                model, tokenizer, question, eps, index.token_ids, cfg,
            )
            eps = [ep for ep, _ in scored[: cfg.top_m]]
        if owns_model:
            del model
            torch.cuda.empty_cache()
        eps.sort(key=lambda e: e.token_range[0])
        texts = [tokenizer.decode(index.token_ids[a:b]) for (a, b) in
                 [(ep.token_range[0], ep.token_range[1]) for ep in eps]]
        ranges = [ep.token_range for ep in eps]
        return texts, ranges

    if cfg.hyde:
        # Generate a short hypothetical-passage continuation, then embed
        # the full (template + question + hypothetical passage) sequence.
        # Pads with eos for tokenizers without a dedicated pad token.
        prompt = cfg.hyde_template.format(question=question)
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(cfg.indexer_device)
        eos = tokenizer.eos_token_id or 0
        with torch.no_grad():
            gen = model.generate(
                prompt_ids,
                max_new_tokens=cfg.hyde_max_tokens,
                do_sample=False,
                pad_token_id=eos,
            )
        q_ids = gen
    else:
        q_ids = tokenizer.encode(question, return_tensors="pt").to(cfg.indexer_device)
    with torch.no_grad():
        q_out = model(q_ids, output_hidden_states=True, use_cache=False)
    q_hidden = _select_hidden_chunk(q_out.hidden_states, cfg.indexer_layer)
    q_emb = _pool_hidden(q_hidden, cfg.query_pool, cfg.pool_alpha)

    # Cosine top-K (wider when re-ranking)
    cosine_k = cfg.rerank_topk if cfg.rerank else cfg.top_m
    top = index.store.topk(q_emb, k=cosine_k, metric="cosine")
    eps: list[Episode] = [ep for ep, _ in top]

    if cfg.rerank:
        # Decode candidates, score them with the indexer model, keep top-M.
        scored = _rerank_with_model(
            model, tokenizer, question, eps, index.token_ids, cfg,
        )
        eps = [ep for ep, _ in scored[: cfg.top_m]]

    if owns_model:
        del model
        torch.cuda.empty_cache()

    # Sort selected episodes in causal (token-position) order before concatenation
    eps.sort(key=lambda e: e.token_range[0])
    texts = [tokenizer.decode(index.token_ids[a:b]) for (a, b) in
             [(ep.token_range[0], ep.token_range[1]) for ep in eps]]
    ranges = [ep.token_range for ep in eps]
    return texts, ranges


def _rerank_with_model(
    model, tokenizer, question: str,
    candidates: list[Episode], doc_token_ids: list[int],
    cfg: EMRAGConfig,
) -> list[tuple[Episode, float]]:
    """LLM-as-reranker: for each candidate episode, score the model's
    yes/no judgment of whether the answer to ``question`` lies in it.

    Returns ``[(Episode, score), ...]`` sorted by descending score.
    Score = logit(" yes") − logit(" no") at the answer position.

    Episodes longer than ``cfg.rerank_max_episode_chars`` are truncated
    in character space (cheap, no re-tokenization for measurement).
    """
    # Resolve yes/no token ids once.
    yes_ids = tokenizer.encode(" yes", add_special_tokens=False)
    no_ids = tokenizer.encode(" no", add_special_tokens=False)
    yes_id = yes_ids[0] if yes_ids else tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = no_ids[0] if no_ids else tokenizer.encode("no", add_special_tokens=False)[0]

    scores: list[tuple[Episode, float]] = []
    for ep in candidates:
        a, b = ep.token_range
        ep_text = tokenizer.decode(doc_token_ids[a:b])
        if len(ep_text) > cfg.rerank_max_episode_chars:
            # Keep head + tail; needle usually lives in the original episode
            # span so truncating evenly is a reasonable default.
            head = cfg.rerank_max_episode_chars // 2
            tail = cfg.rerank_max_episode_chars - head
            ep_text = ep_text[:head] + " […] " + ep_text[-tail:]
        prompt = cfg.rerank_template.format(episode=ep_text, question=question)
        ids = tokenizer.encode(prompt, return_tensors="pt").to(cfg.indexer_device)
        with torch.no_grad():
            out = model(ids, use_cache=False)
        next_logits = out.logits[0, -1]
        score = float((next_logits[yes_id] - next_logits[no_id]).item())
        scores.append((ep, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def em_rag(question: str, document: str, cfg: EMRAGConfig | None = None,
           max_tokens: int = 512) -> dict:
    """End-to-end: index the document, retrieve, generate.

    Returns a dict with the answer plus retrieval metadata so callers
    can audit which spans were consulted.
    """
    cfg = cfg or EMRAGConfig()
    print(f"[em_rag] loading indexer ({cfg.indexer_model}) ...", flush=True)
    model, tokenizer = _load_indexer(cfg)
    try:
        print(f"[em_rag] indexing {len(document):,} chars ...", flush=True)
        index = build_index(document, cfg, model=model, tokenizer=tokenizer)
        print(f"[em_rag] {index.store.num_episodes} episodes, "
              f"retrieving top-{cfg.top_m}", flush=True)
        texts, ranges = retrieve_episode_texts(
            index, question, cfg, model=model, tokenizer=tokenizer)
    finally:
        del model
        torch.cuda.empty_cache()

    print("[em_rag] generating with AWQ ...", flush=True)
    # Lazy import vLLM so we don't hold both runtimes resident at once.
    from .vanilla import VanillaConfig, VanillaSession
    awq_cfg = VanillaConfig()
    session = VanillaSession(awq_cfg)
    context = "\n\n---\n\n".join(texts)
    answer = session.query(
        question=question,
        document=context,
        max_tokens=max_tokens,
    )
    return {
        "answer": answer,
        "num_episodes": index.store.num_episodes,
        "retrieved_ranges": ranges,
        "retrieved_chars": sum(len(t) for t in texts),
    }
