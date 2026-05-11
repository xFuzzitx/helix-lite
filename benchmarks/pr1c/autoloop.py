"""Autonomous PR1c iteration runner.

Loops over calibration variants until one passes both:
  (a) validate_scales.py NIAH (closed-loop hook test on HF Qwen2)
  (b) vLLM Phase 1A smoke at 4K, 32K, 128K, 256K

On each round, writes a JSON status to ``benchmarks/results/pr1c_iter_log.json``
and a per-round log to ``logs/pr1c_round_<n>.log``.

Round 0  validate already-on-disk nuq2_v3 (currently calibrating)
Round 1  mix nuq2_v3 with nuq4_v3 at cut=20
Round 2  mix at cut=16
Round 3  fresh nuq2 calib with outlier_pct=0.08, seqlen=4096
Round 4  fresh nuq2 calib with outlier_pct=0.05, seqlen=16384 (long-context)

The runner reads existing on-disk artefacts when possible to skip
already-run rounds. Hard-fails after 5 rounds without a pass.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "benchmarks" / "results"
SCALES_DIR = ROOT / "scales"
ENV_BASE = {
    "HF_HOME": os.environ.get("HF_HOME", "/snapshots/helix-lite/hf-cache"),
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONPATH": str(ROOT / "src"),
    "CUDA_VISIBLE_DEVICES": "1",
}
PY = "/snapshots/helix-lite/.venv/bin/python"


def now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run(cmd: list[str], log_path: Path, timeout: int = 7200) -> tuple[int, str]:
    """Run cmd, tee to log_path, return (rc, tail)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **ENV_BASE}
    print(f"$ {' '.join(cmd)}\n  → {log_path}", flush=True)
    with log_path.open("w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            return -9, f"TIMEOUT after {timeout}s"
    tail = log_path.read_text(errors="replace").splitlines()[-50:]
    return rc, "\n".join(tail)


def validate_scales(scales_path: Path, round_name: str) -> dict:
    """Run validate_scales.py and parse the verdict.

    Returns dict with keys: 'passed', 'baseline_needle', 'q_needle',
    'log_path', 'l27_v_mean'.
    """
    log = LOG_DIR / f"pr1c_{round_name}_validate.log"
    cmd = [PY, "-m", "kvquant.validate_scales", "--scales", str(scales_path),
           "--target-tokens", "2000"]
    rc, tail = run(cmd, log, timeout=900)
    text = log.read_text(errors="replace") if log.exists() else ""

    base = re.search(r"baseline needle:\s*(YES|NO)", text)
    quan = re.search(r"nuq2 needle:\s*(YES|NO)", text) or re.search(r"quantised needle:\s*(YES|NO)", text)
    # Layer-27 V mean error: find Value-reconstruction line for layer 27
    m_v = re.search(r"\n   27\s+([\d.]+)\s+", text.split("Value reconstruction:")[-1])
    return {
        "rc": rc,
        "log_path": str(log),
        "baseline_needle": base.group(1) if base else "?",
        "q_needle": quan.group(1) if quan else "?",
        "passed": (rc == 0 and quan and quan.group(1) == "YES"),
        "l27_v_mean": float(m_v.group(1)) if m_v else None,
    }


def vllm_smoke(scales_path: Path, round_name: str,
               ctxs: list[int] = (4000, 32000, 128000, 256000),
               max_model_len: int = 260000,
               gmu: float = 0.92) -> dict:
    """Run the Phase 1A vLLM smoke at increasing contexts; parse PASS/FAIL."""
    log = LOG_DIR / f"pr1c_{round_name}_smoke.log"
    cmd = [PY, str(ROOT / "benchmarks" / "pr1c" / "smoke_nuq_vllm.py"),
           "--scales", str(scales_path), "--gmu", str(gmu),
           "--max-tokens", "32", "--max-model-len", str(max_model_len),
           "--ctx", *[str(c) for c in ctxs]]
    rc, tail = run(cmd, log, timeout=3600)
    text = log.read_text(errors="replace") if log.exists() else ""
    per_ctx = []
    for ctx in ctxs:
        m = re.search(rf"ctx={ctx:,} \(actual ([\d,]+) tok\).*?({chr(0x2713)}|{chr(0x2717)}) elapsed=([\d.]+)s", text, re.S)
        if m:
            per_ctx.append({
                "ctx_target": ctx,
                "ctx_actual": int(m.group(1).replace(",", "")),
                "pass": m.group(2) == chr(0x2713),
                "elapsed_s": float(m.group(3)),
            })
        else:
            per_ctx.append({"ctx_target": ctx, "pass": False, "elapsed_s": None})
    overall = (rc == 0 and "=== PASS ===" in text)
    return {
        "rc": rc, "log_path": str(log),
        "overall_pass": overall, "per_ctx": per_ctx,
    }


def calibrate(out_path: Path, *, num_bits: int = 2, outlier_pct: float = 0.05,
              seqlen: int = 4096, num_prompts: int = 32,
              kmeans_subsample: int = 5000) -> dict:
    """Run a fresh calibration. Returns dict with 'rc', 'log_path', 'duration_s'."""
    name = out_path.stem
    log = LOG_DIR / f"pr1c_calib_{name}.log"
    cmd = [PY, "-m", "kvquant.calibration",
           "--num-prompts", str(num_prompts),
           "--seqlen", str(seqlen),
           "--num-bits", str(num_bits),
           "--outlier-pct", str(outlier_pct),
           "--kmeans-subsample", str(kmeans_subsample),
           "--out", str(out_path),
           "--progress-log", str(SCALES_DIR / f"calibration_{name}.log")]
    t0 = time.time()
    rc, _tail = run(cmd, log, timeout=14400)  # 4h cap
    return {"rc": rc, "log_path": str(log), "duration_s": time.time() - t0,
            "exists": out_path.exists()}


def mix_scales(shallow: Path, deep: Path, cut: int, out: Path) -> dict:
    cmd = [PY, str(ROOT / "benchmarks" / "pr1c" / "mix_scales.py"),
           "--shallow", str(shallow), "--deep", str(deep),
           "--cut", str(cut), "--out", str(out)]
    log = LOG_DIR / f"pr1c_mix_{out.stem}.log"
    rc, _ = run(cmd, log, timeout=120)
    return {"rc": rc, "log_path": str(log), "exists": out.exists()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--skip-smoke", action="store_true",
                   help="run validate_scales only, don't fire vLLM smoke")
    p.add_argument("--smoke-ctxs", type=int, nargs="+",
                   default=[4000, 32000, 128000, 256000])
    args = p.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    status_path = RESULTS_DIR / "pr1c_iter_log.json"
    status: list[dict] = []
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
        except Exception:
            status = []

    def save_status():
        status_path.write_text(json.dumps(status, indent=2))

    rounds = [
        # name, scales path, generator callable returning the path
        ("nuq2_v3", SCALES_DIR / "qwen2_5_7b_1m_nuq2_v3.pt",
         lambda: SCALES_DIR / "qwen2_5_7b_1m_nuq2_v3.pt"),  # already calibrating
        ("mixed_v3_cut20", SCALES_DIR / "mixed_nuq2v3_nuq4v3_cut20.pt",
         lambda: _mix(SCALES_DIR / "qwen2_5_7b_1m_nuq2_v3.pt",
                      SCALES_DIR / "qwen2_5_7b_1m_nuq4_v3.pt", 20,
                      SCALES_DIR / "mixed_nuq2v3_nuq4v3_cut20.pt")),
        ("mixed_v3_cut16", SCALES_DIR / "mixed_nuq2v3_nuq4v3_cut16.pt",
         lambda: _mix(SCALES_DIR / "qwen2_5_7b_1m_nuq2_v3.pt",
                      SCALES_DIR / "qwen2_5_7b_1m_nuq4_v3.pt", 16,
                      SCALES_DIR / "mixed_nuq2v3_nuq4v3_cut16.pt")),
        ("mixed_v3_cut24", SCALES_DIR / "mixed_nuq2v3_nuq4v3_cut24.pt",
         lambda: _mix(SCALES_DIR / "qwen2_5_7b_1m_nuq2_v3.pt",
                      SCALES_DIR / "qwen2_5_7b_1m_nuq4_v3.pt", 24,
                      SCALES_DIR / "mixed_nuq2v3_nuq4v3_cut24.pt")),
        ("nuq2_v4_ol08", SCALES_DIR / "qwen2_5_7b_1m_nuq2_v4_ol08.pt",
         lambda: _calib(SCALES_DIR / "qwen2_5_7b_1m_nuq2_v4_ol08.pt",
                        num_bits=2, outlier_pct=0.08, seqlen=4096,
                        kmeans_subsample=3000)),
    ]

    found = None
    for i, (name, path, gen) in enumerate(rounds[: args.max_rounds]):
        print(f"\n========== Round {i}: {name} ==========", flush=True)
        round_entry = {"round": i, "name": name, "scales": str(path),
                       "ts": now(), "validate": None, "smoke": None}

        # Ensure scales exist
        if not path.exists():
            print(f"[round {i}] scales {path} missing, generating ...", flush=True)
            gen()
        if not path.exists():
            round_entry["error"] = f"failed to produce {path}"
            status.append(round_entry); save_status()
            print(f"[round {i}] generation failed, skipping", flush=True)
            continue

        v = validate_scales(path, name)
        round_entry["validate"] = v
        print(f"[round {i}] validate: passed={v['passed']} "
              f"q_needle={v['q_needle']} l27_v_mean={v['l27_v_mean']}", flush=True)
        status.append(round_entry); save_status()

        if not v["passed"]:
            print(f"[round {i}] FAIL on validate, advancing", flush=True)
            continue

        if args.skip_smoke:
            print(f"[round {i}] validate passed, --skip-smoke set, stopping",
                  flush=True)
            found = round_entry; break

        s = vllm_smoke(path, name, ctxs=args.smoke_ctxs)
        round_entry["smoke"] = s
        save_status()
        print(f"[round {i}] vllm smoke: overall_pass={s['overall_pass']}",
              flush=True)
        for c in s["per_ctx"]:
            print(f"  ctx={c['ctx_target']:>7}  pass={c.get('pass')}  "
                  f"elapsed={c.get('elapsed_s')}", flush=True)

        if s["overall_pass"]:
            found = round_entry
            break
        print(f"[round {i}] FAIL on smoke, advancing", flush=True)

    if found:
        print(f"\n=== WINNING ROUND {found['round']}: {found['name']} ===",
              flush=True)
        print(json.dumps(found, indent=2))
        return 0
    print("\n=== NO ROUND PASSED ===", flush=True)
    return 1


def _calib(out_path: Path, **kwargs):
    res = calibrate(out_path, **kwargs)
    print(f"  calibrate rc={res['rc']} dur={res['duration_s']:.0f}s exists={res['exists']}")
    return out_path


def _mix(shallow: Path, deep: Path, cut: int, out: Path):
    res = mix_scales(shallow, deep, cut, out)
    print(f"  mix rc={res['rc']} exists={res['exists']}")
    return out


if __name__ == "__main__":
    sys.exit(main())
