"""HELIX-Lite — queryable long-context inference on 2x RTX 3090.

Two query modes are exposed:

* :func:`vanilla` - load Qwen2.5-7B-Instruct-1M-AWQ via vLLM and
  answer a question with the full document in context. Caps at 128K
  tokens on a single 3090.

* :func:`em_rag` - segment the document, build a max-abs-pooled
  EpisodeStore on GPU 1, retrieve the top-M most relevant episodes
  for the question, and answer over the retrieved chunks plus the
  question. Scales past 128K because only ``hot + M*episode_len``
  tokens enter the model at inference time.

Run from the command line::

    python -m helix --doc large.txt "What is the secret password?"
    python -m helix --doc 5M_book.txt --em-rag --top-m 16 "..."
"""
from .vanilla import vanilla
from .em_rag import em_rag, EpisodeIndex, build_index

__all__ = ["vanilla", "em_rag", "EpisodeIndex", "build_index"]
