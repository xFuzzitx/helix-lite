"""HELIX-Lite command-line entry point.

Examples::

    python -m helix "What is the capital of France?"
    python -m helix --doc large.txt "What is the secret password?"
    python -m helix --doc 5M_book.txt --em-rag --top-m 16 \\
        "Who killed Roger Ackroyd?"
    python -m helix --repl --doc large.txt    # interactive multi-turn

The first run downloads the AWQ weights (~3.5 GB) plus the indexer
model when ``--em-rag`` is used.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(prog="helix", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="?",
                   help="question to ask. Omit with --repl for interactive mode.")
    p.add_argument("--doc", type=Path,
                   help="path to a document to ground the question in")
    p.add_argument("--em-rag", action="store_true",
                   help="use EM-LLM-style retrieval (recommended for docs > 128K tokens)")
    p.add_argument("--top-m", type=int, default=16,
                   help="number of episodes to retrieve when --em-rag")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="max tokens to generate (default 512)")
    p.add_argument("--max-doc-tokens", type=int, default=200_000,
                   help="cap on indexed document length when --em-rag")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--repl", action="store_true", help="interactive multi-turn shell")
    p.add_argument("--nuq-scales", type=str, default=None,
                   help="optional KVScales .pt path. Installs the PR1c Phase 1A "
                        "math-wrapper so every K/V round-trips through nuq "
                        "quant→dequant. Validates calibration at scale; the "
                        "fp16 KV pool is unchanged so no VRAM savings yet. "
                        "Winning recipe: scales/mixed_nuq2v3_nuq4v3_cut16.pt")
    p.add_argument("--max-model-len", type=int, default=128_000)
    p.add_argument("--gmu", type=float, default=0.85,
                   help="gpu_memory_utilization for vLLM (default 0.85; "
                        "bump to 0.92 if using a single dedicated GPU)")
    args = p.parse_args()

    document = None
    if args.doc:
        if not args.doc.exists():
            print(f"error: {args.doc} does not exist", file=sys.stderr)
            return 2
        document = args.doc.read_text(encoding="utf-8", errors="replace")

    if args.repl:
        return _run_repl(document, args)

    if not args.question:
        p.error("provide a question, or use --repl")

    return _run_one(args.question, document, args)


def _run_one(question: str, document: str | None, args: argparse.Namespace) -> int:
    if args.em_rag and document is not None:
        from .em_rag import em_rag, EMRAGConfig
        cfg = EMRAGConfig(top_m=args.top_m, max_doc_tokens=args.max_doc_tokens)
        result = em_rag(question, document, cfg=cfg, max_tokens=args.max_tokens)
        print(f"\n[retrieved {len(result['retrieved_ranges'])} episodes covering "
              f"{result['retrieved_chars']:,} chars]")
        print(f"\n{result['answer']}")
    else:
        from .vanilla import VanillaSession, VanillaConfig
        session = VanillaSession(VanillaConfig(
            nuq_scales=args.nuq_scales,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gmu,
        ))
        ans = session.query(question, document,
                             max_tokens=args.max_tokens,
                             temperature=args.temperature)
        print(ans)
    return 0


def _run_repl(document: str | None, args: argparse.Namespace) -> int:
    """Interactive multi-turn shell. Loads the model once, answers
    each question against the same document. ``/quit`` to exit."""
    from .vanilla import VanillaSession, VanillaConfig

    print("HELIX-Lite REPL. Type your question and hit enter; '/quit' to exit.")
    if document is not None and len(document) > 0:
        print(f"loaded document: {len(document):,} chars")
    print("(loading model, ~30s the first time) ...")
    session = VanillaSession(VanillaConfig())
    print("ready.")

    while True:
        try:
            q = input("\nhelix> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q in ("/quit", "/exit", "/q"):
            return 0
        ans = session.query(q, document,
                             max_tokens=args.max_tokens,
                             temperature=args.temperature)
        print(ans)


if __name__ == "__main__":
    sys.exit(main())
