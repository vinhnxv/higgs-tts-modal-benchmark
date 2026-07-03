"""Benchmark client for the Higgs TTS Modal deployment.

Two invocation styles:

1. Co-run with an ephemeral server (cold-start + sweep, one tier at a time):
       HIGGS_TIER=L4 modal run benchmark_client.py::benchmark --gpu-type L4
       HIGGS_TIER=H100 modal run benchmark_client.py::benchmark --gpu-type H100 \\
           --concurrency-levels 1,4,8,16 --pattern both

2. Drive a DEPLOYED server by URL (snapshot test, external benchmark):
       modal_deploy higgs_modal.py ; then capture the URL and:
       HIGGS_TIER=L4 modal run benchmark_client.py::benchmark --gpu-type L4 \\
           --url https://...modal.run --mode snapshot --pattern zero-shot

Local (no Modal provisioning) analysis:
       python benchmark_client.py summarize        # results/*.json -> summary.json
       python benchmark_client.py breakeven        # summary + results -> breakeven.json

A pure-stdlib HTTP client (urllib + ThreadPoolExecutor) is used on the LOCAL side
so the runner needs no third-party packages beyond `modal` itself.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Importing higgs_modal is harmless for local analyze (no remote calls fire);
# it gives us the modal app + HiggsTTS class + tier config for the entrypoint.
from higgs_modal import TIER as ENV_TIER, GPU as ENV_GPU  # noqa: F401
import modal  # noqa: E402,F401
from higgs_modal import app, HiggsTTS  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# --- constants -------------------------------------------------------------
MODEL = "bosonai/higgs-audio-v3-tts-4b"
REF_AUDIO = "/ref_audio/ENG_UK_M_DaveB.wav"
REF_TEXT = (
    "Sodi Scientifica has been designing and marketing traffic enforcement "
    "systems for nearly fifty years, with the goal of improving road safety "
    "and people's quality of life. As an internationally recognised leader in "
    "the field, Sodi Scientifica has always invested in research and "
    "development with the aim of introducing technologically advanced "
    "solutions."
)
ZERO_SHOT_PROMPTS: list[tuple[str, str]] = [
    ("en", "Hello, how are you?"),
    ("vi", "Xin chào, bạn có khỏe không?"),
]
VOICE_CLONE_INPUT = "Have a nice day and enjoy south california sunshine."

GPU_PRICES_USD_HR: dict[str, float] = {
    "L4": 0.80,
    "A10": 1.10,
    "L40S": 1.95,
    "A100_40": 2.10,
    "H100": 3.95,
}
DEFAULT_CONCURRENCY_LEVELS = [1, 4, 8, 16]
REQUEST_TIMEOUT_S = 900  # cold starts can exceed the model load (~20 min)


# --- HTTP (stdlib) ---------------------------------------------------------
def _post_audio(url: str, body: dict, timeout: float = REQUEST_TIMEOUT_S) -> tuple[float, bytes]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/audio/speech",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ct = r.headers.get("Content-Type", "")
        body_bytes = r.read()
    dt = time.time() - t0
    # Validate response is audio, not an HTML error page.
    if "audio" not in ct and len(body_bytes) < 1000:
        raise ValueError(f"Expected audio response, got Content-Type={ct}, body={body_bytes[:200]}")
    return dt, body_bytes


def _health(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def _parse_wav_duration(b: bytes) -> float:
    """Return duration in seconds for a WAV blob, best-effort."""
    try:
        with wave_open_bytes(b) as wf:
            n = wf.getnframes()
            sr = wf.getframerate() or 24000
            return n / sr
    except Exception:
        return 0.0


def wave_open_bytes(b: bytes):
    import wave

    return wave.open(io.BytesIO(b))


# --- request bodies --------------------------------------------------------
def _zeroshot_body() -> dict:
    # Use the English prompt for the sweep (stable, comparable across tiers).
    return {"input": ZERO_SHOT_PROMPTS[0][1], "response_format": "wav"}


def _voiceclone_body() -> dict:
    return {
        "input": VOICE_CLONE_INPUT,
        "response_format": "wav",
        "references": [{"audio_path": REF_AUDIO, "text": REF_TEXT}],
        "temperature": 0.8,
        "top_k": 50,
        "max_new_tokens": 1024,
    }


def _build_body(pattern: str) -> dict:
    if pattern == "zero-shot":
        return _zeroshot_body()
    if pattern == "voice-cloning":
        return _voiceclone_body()
    raise ValueError(f"unknown pattern {pattern!r}")


# --- measurement primitives -----------------------------------------------
def _one_request(url: str, body: dict) -> dict:
    try:
        dt, b = _post_audio(url, body, timeout=REQUEST_TIMEOUT_S)
        dur = _parse_wav_duration(b)
        return {
            "ok": True,
            "latency_s": round(dt, 3),
            "audio_duration_s": round(dur, 3),
            "bytes": len(b),
            "rtf": round(dt / dur, 3) if dur > 0 else None,
        }
    except urllib.error.HTTPError as e:
        msg = f"HTTP {e.code}"
        try:
            err_body = e.read().decode("utf-8", errors="ignore")[:400]
        except Exception:
            err_body = ""
        return {"ok": False, "latency_s": None, "audio_duration_s": None, "bytes": 0,
                "rtf": None, "error": msg, "err_body": err_body}
    except Exception as e:
        return {"ok": False, "latency_s": None, "audio_duration_s": None, "bytes": 0,
                "rtf": None, "error": f"{type(e).__name__}: {e}"}


def _sweep_concurrency(url: str, n: int, body: dict) -> dict:
    bodies = [body] * n
    results: list[dict] = [None] * n  # type: ignore[list-item]
    t_send = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futs = {pool.submit(_one_request, url, b): i for i, b in enumerate(bodies)}
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                              "latency_s": None, "audio_duration_s": None, "bytes": 0, "rtf": None}
    wall = time.time() - t_send
    oks = [r for r in results if r["ok"]]
    lats = [r["latency_s"] for r in oks if r["latency_s"] is not None]
    durs = [r["audio_duration_s"] for r in oks if r["audio_duration_s"] is not None]
    errs = [r for r in results if not r["ok"]]
    out = {
        "concurrency": n,
        "ok": len(oks),
        "failed": len(errs),
        "wall_s": round(wall, 3),
        "throughput_req_s": round(len(oks) / wall, 3) if wall > 0 else 0.0,
        "mean_latency_s": round(statistics.mean(lats), 3) if lats else None,
        "p50_latency_s": round(statistics.median(lats), 3) if lats else None,
        "max_latency_s": round(max(lats), 3) if lats else None,
        "mean_audio_duration_s": round(statistics.mean(durs), 3) if durs else None,
        "total_audio_s": round(sum(durs), 3) if durs else 0.0,
        "rtf": (
            round(statistics.mean(lats) / statistics.mean(durs), 3)
            if durs and lats and statistics.mean(durs) > 0
            else None
        ),
        "audio_s_per_s": round(sum(durs) / wall, 3) if wall > 0 and durs else 0.0,
        "errors": errs[:5],
    }
    return out


def _measure_cold(url: str, body: dict) -> dict:
    """Single request that includes container init + model load + inference."""
    res = _one_request(url, body)
    if res["ok"]:
        return {"cold_start_s": res["latency_s"], "audio_duration_s": res["audio_duration_s"], "ok": True,
                "bytes": res["bytes"]}
    return {"cold_start_s": None, "ok": False, "error": res.get("error"),
            "err_body": res.get("err_body")}


# --- benchmark driver ------------------------------------------------------
PATTERNS_ALL = ["zero-shot", "voice-cloning"]


def run_benchmark(
    url: str,
    gpu_type: str,
    mode: str,
    pattern: str,
    levels: list[int],
    write_dir: Path = RESULTS,
) -> list[str]:
    """Run the cold/warm benchmark for one tier and one or both patterns. Returns output paths."""
    price = GPU_PRICES_USD_HR.get(gpu_type)
    meta = {
        "tier": gpu_type,
        "gpu_requested": ENV_GPU,
        "price_usd_per_hr": price,
        "mode": mode,
        "concurrency_levels": levels,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    written: list[str] = []

    patterns = PATTERNS_ALL if pattern == "both" else [pattern]
    for pat in patterns:
        body = _build_body(pat)
        result: dict = {**meta, "pattern": pat}

        if mode == "cold":
            print(f"[{gpu_type}/{pat}] measuring cold start...", flush=True)
            cold = _measure_cold(url, body)
            result["cold_start"] = cold
            if not cold.get("ok"):
                # Server failed outright on first request — log and stop this pattern.
                print(f"[{gpu_type}/{pat}] COLD FAILED: {cold.get('error')} "
                      f"{(cold.get('err_body') or '')[:200]}", flush=True)
                _write_result(write_dir, gpu_type, mode, pat, result)
                written.append(_result_path(write_dir, gpu_type, mode, pat))
                continue
            print(f"[{gpu_type}/{pat}] cold_start={cold['cold_start_s']}s "
                  f"audio={cold['audio_duration_s']}s", flush=True)
            # extra warmup throw-away before sweep
            try:
                _one_request(url, body)
            except Exception:
                pass
        elif mode == "snapshot":
            # For deployed snapshot server: measure single-request cold (snapshot restore)
            print(f"[{gpu_type}/{pat}] measuring snapshot cold start...", flush=True)
            cold = _measure_cold(url, body)
            result["snapshot_cold_start"] = cold
            print(f"[{gpu_type}/{pat}] snapshot_cold_start={cold.get('cold_start_s')}s", flush=True)
            # use one warm throw-away before sweep
            try:
                _one_request(url, body)
            except Exception:
                pass
        else:  # warm
            print(f"[{gpu_type}/{pat}] warm mode — health probe...", flush=True)
            for _ in range(60):
                if _health(url):
                    break
                time.sleep(5)
            try:
                _one_request(url, body)
            except Exception:
                pass

        sweep_results: list[dict] = []
        for n in levels:
            print(f"[{gpu_type}/{pat}] sweep concurrency={n}...", flush=True)
            row = _sweep_concurrency(url, n, body)
            print(
                f"  ok={row['ok']} fail={row['failed']} wall={row['wall_s']}s "
                f"thr={row['throughput_req_s']} req/s mean_lat={row['mean_latency_s']}s "
                f"rtf={row['rtf']} audio_s/s={row['audio_s_per_s']}",
                flush=True,
            )
            sweep_results.append(row)
            # Stop escalating concurrency if everything is failing (likely OOM).
            if row["ok"] == 0 and n >= 4:
                print(f"[{gpu_type}/{pat}] all failed at N={n}; stopping sweep escalation", flush=True)
                break
        result["sweep"] = sweep_results
        _write_result(write_dir, gpu_type, mode, pat, result)
        rel = _result_path(write_dir, gpu_type, mode, pat)
        written.append(rel)
        print(f"[{gpu_type}/{pat}] wrote {rel}", flush=True)

    return written


def _result_path(write_dir: Path, gpu_type: str, mode: str, pattern: str) -> str:
    return str((write_dir / f"{gpu_type}_{mode}_{pattern}.json").relative_to(HERE))


def _write_result(write_dir: Path, gpu_type: str, mode: str, pattern: str, result: dict) -> None:
    p = write_dir / f"{gpu_type}_{mode}_{pattern}.json"
    p.write_text(json.dumps(result, indent=2))


# --- Modal entrypoint (ephemeral co-run) -----------------------------------
@app.local_entrypoint()
def benchmark(
    url: str | None = None,
    mode: str = "cold",
    pattern: str = "both",
    concurrency_levels: str = "1,4,8,16",
    gpu_type: str | None = None,
) -> str:
    """Co-run with an ephemeral HiggsTTS server (one tier) and run the benchmark."""
    if gpu_type is None:
        gpu_type = ENV_TIER
    if gpu_type != ENV_TIER:
        raise SystemExit(
            f"--gpu-type {gpu_type!r} must match HIGGS_TIER={ENV_TIER!r}. "
            f"Re-run with: HIGGS_TIER={gpu_type} modal run benchmark_client.py::benchmark --gpu-type {gpu_type}"
        )
    levels = [int(x) for x in concurrency_levels.split(",") if x.strip()]
    if url is None:
        # Provision an ephemeral server container on this tier.
        url = HiggsTTS(tier=gpu_type).serve.get_web_url()
    print(f"[benchmark] tier={gpu_type} url={url} mode={mode} pattern={pattern} levels={levels}", flush=True)
    files = run_benchmark(url, gpu_type, mode, pattern, levels)
    return "\n".join(files)


def run_snapshot_test(deployed_app_name: str, snapshot_tier: str = "L4") -> dict:
    """Probe a DEPLOYED snapshot server for compatibility + best-effort reduction.

    Compatible = /release_memory_occupation responded 2xx during the container's
    snapshot-phase startup. The cold sample is one request served from the
    deployed snapshot server; methodology documented in the verdict JSON.
    """
    cls = modal.Cls.from_name(deployed_app_name, "HiggsTTS")
    status = cls().snapshot_endpoint_status.remote()
    url = cls().serve.get_web_url()

    compatible = bool(status.get("compatible"))
    print(f"[snapshot] status={status} url={url}", flush=True)

    cold_snapshot = None
    failure_reason = None
    if compatible:
        try:
            # Force a fresh container (snapshot-restore) by stopping a warm one first.
            try:
                _one_request(url, _build_body("zero-shot"))  # ensure server is up / snapshotted
            except Exception:
                pass
            time.sleep(2)
            res = _one_request(url, _build_body("zero-shot"))
            cold_snapshot = res.get("latency_s") if res.get("ok") else None
            if res.get("ok") is not True:
                failure_reason = res.get("error")
        except Exception as e:
            failure_reason = f"{type(e).__name__}: {e}"
    else:
        failure_reason = (
            status.get("compatible") is None
            and "snapshot startup phase did not reach /release_memory_occupation"
        ) or ("/release_memory_occupation did not return 2xx")

    # Baseline: reuse the non-snapshot cold value from U5 if present.
    baseline = None
    base_path = RESULTS / f"{snapshot_tier}_cold_zero-shot.json"
    if base_path.exists():
        try:
            base = json.loads(base_path.read_text())
            baseline = base.get("cold_start", {}).get("cold_start_s")
        except Exception:
            pass

    reduction = None
    if compatible and cold_snapshot is not None and baseline not in (None, 0, 0.0):
        reduction = round((baseline - cold_snapshot) / baseline * 100, 1)

    verdict = {
        "compatible": compatible,
        "cold_start_baseline_s": baseline,
        "cold_start_snapshot_s": cold_snapshot,
        "reduction_pct": reduction,
        "failure_reason": None if compatible else failure_reason,
        "tier": snapshot_tier,
        "methodology": "compatible = server-side POST /release_memory_occupation 2xx during "
        "snapshot-phase startup; baseline = U5 non-snapshot cold (zero-shot); "
        "cold_start_snapshot_s = one zero-shot request from the deployed snapshot server "
        "(best-effort restore measure; modal run disables snapshotting so this deploys).",
    }
    (RESULTS / "snapshot_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return verdict


# --- local analysis --------------------------------------------------------
def _load_tier_results() -> list[dict]:
    out: list[dict] = []
    for p in sorted(RESULTS.glob("*.json")):
        if p.name in {"summary.json", "breakeven.json", "snapshot_verdict.json"}:
            continue
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out


def _warm_n1(result: dict) -> dict | None:
    for row in result.get("sweep", []):
        if row.get("concurrency") == 1 and row.get("ok", 0) > 0:
            return row
    return None


def _best_throughput(result: dict) -> dict | None:
    best = None
    for row in result.get("sweep", []):
        if row.get("ok", 0) > 0:
            if best is None or (row.get("throughput_req_s", 0) or 0) > (best.get("throughput_req_s", 0) or 0):
                best = row
    return best


def summarize() -> dict:
    """Aggregate results/*.json into summary.json + print a comparison table.

    Both patterns run in the same container, so the voice-cloning cold_start is
    actually a warm start. The true container cold start is the zero-shot
    cold_start (first request). We use it as the container cold start for all
    patterns in a tier.
    """
    rows = _load_tier_results()
    summary: dict = {"tiers": {}, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    # First pass: collect per-tier raw data and extract the true container cold
    # start from the zero-shot pattern (which runs first and includes model
    # loading). Voice cloning cold_start is a warm start, not a cold start.
    tier_container_cold: dict[str, float | None] = {}
    for r in rows:
        tier = r.get("tier") or r.get("gpu_requested") or "?"
        pat = r.get("pattern", "?")
        if pat == "zero-shot":
            cold = r.get("cold_start") or r.get("snapshot_cold_start") or {}
            tier_container_cold[tier] = cold.get("cold_start_s")

    # Second pass: compute cost metrics using the true container cold start.
    for r in rows:
        tier = r.get("tier") or r.get("gpu_requested") or "?"
        tier_block = summary["tiers"].setdefault(
            tier, {"price_usd_per_hr": r.get("price_usd_per_hr"), "patterns": {}}
        )
        pat = r.get("pattern", "?")
        cold = r.get("cold_start") or r.get("snapshot_cold_start") or {}
        w1 = _warm_n1(r)
        best = _best_throughput(r)
        pattern_cold_s = cold.get("cold_start_s")  # per-pattern (warm for voice-cloning)
        container_cold_s = tier_container_cold.get(tier)  # true container cold start
        processing_s = w1.get("mean_latency_s") if w1 else None
        audio_dur = (w1 or {}).get("mean_audio_duration_s")
        total_audio = sum((row.get("total_audio_s") or 0) for row in r.get("sweep", []))
        rate = GPU_PRICES_USD_HR.get(tier, 0.0)
        # Cost per request uses the true container cold start (zero-shot) for
        # all patterns, not the per-pattern warm-start cold.
        cold_for_cost = container_cold_s if container_cold_s is not None else pattern_cold_s
        cost_per_req_cold = (
            round(((cold_for_cost or 0) + (processing_s or 0)) * (rate / 3600), 4)
            if cold_for_cost is not None and processing_s is not None
            else None
        )
        cost_per_audio_cold = (
            round(((cold_for_cost or 0) + (processing_s or 0)) * (rate / 3600) / audio_dur, 4)
            if cold_for_cost is not None and processing_s is not None and audio_dur
            else None
        )
        # warm steady-state per-request cost (no cold)
        cost_per_req_warm = (
            round((processing_s or 0) * (rate / 3600), 4) if processing_s is not None else None
        )
        tier_block["patterns"][pat] = {
            "cold_start_s": pattern_cold_s,
            "container_cold_start_s": container_cold_s,
            "is_warm_start": pat == "voice-cloning" and container_cold_s is not None,
            "warm_n1_processing_s": processing_s,
            "warm_n1_audio_s": audio_dur,
            "warm_n1_rtf": (w1 or {}).get("rtf"),
            "best_throughput_req_s": (best or {}).get("throughput_req_s"),
            "best_throughput_concurrency": (best or {}).get("concurrency"),
            "mean_latency_at_best_s": (best or {}).get("mean_latency_s"),
            "total_audio_s_sweep": round(total_audio, 3),
            "cost_per_req_cold_usd": cost_per_req_cold,
            "cost_per_audio_sec_cold_usd": cost_per_audio_cold,
            "cost_per_req_warm_usd": cost_per_req_warm,
        }

    # Pick sweet spot: lowest cost_per_audio_sec_cold_usd using true container
    # cold starts. Voice cloning warm-start cold_s is excluded from the sweet
    # spot calculation since it doesn't represent a real cold start.
    scored = []
    for tier, tb in summary["tiers"].items():
        for pat, pb in tb["patterns"].items():
            c = pb.get("cost_per_audio_sec_cold_usd")
            if c is not None:
                scored.append((c, tier, pat, pb))
    scored.sort(key=lambda x: x[0])
    if scored:
        c, tier, pat, pb = scored[0]
        summary["sweet_spot"] = {
            "tier": tier,
            "pattern": pat,
            "cost_per_audio_sec_cold_usd": c,
            "cost_per_req_cold_usd": pb.get("cost_per_req_cold_usd"),
            "reasoning": (
                "lowest cost per audio second (cold-inclusive, using true container "
                "cold start from zero-shot first request) across valid tier/pattern runs; "
                "primary metric for bursty scale-to-zero traffic."
            ),
        }
    else:
        summary["sweet_spot"] = None

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    _print_summary_table(summary)
    return summary


def _print_summary_table(summary: dict) -> None:
    print("\n=== Higgs TTS Modal Benchmark — Cost Comparison ===")
    header = f"{'Tier':<9}{'$/hr':>5}{'Pat':<13}{'ColdS':>7}{'ProcS':>7}{'Thr(req/s)':>11}{'$/reqCold':>10}{'$/audioS':>9}"
    print(header)
    print("-" * len(header))
    for tier in sorted(summary["tiers"]):
        tb = summary["tiers"][tier]
        rate = tb.get("price_usd_per_hr")
        for pat, pb in sorted(tb["patterns"].items()):
            print(
                f"{tier:<9}{(rate or 0):>5.2f}{pat:<13}"
                f"{_f(pb.get('cold_start_s')):>7}{_f(pb.get('warm_n1_processing_s')):>7}"
                f"{_f(pb.get('best_throughput_req_s')):>11}"
                f"{_f(pb.get('cost_per_req_cold_usd'), 4):>10}"
                f"{_f(pb.get('cost_per_audio_sec_cold_usd'), 4):>9}"
            )
    if summary.get("sweet_spot"):
        ss = summary["sweet_spot"]
        print(
            f"\nSWEET SPOT (bursty scale-to-zero): {ss['tier']} ({ss['pattern']}) "
            f"cost/audio_sec(cold)=${ss['cost_per_audio_sec_cold_usd']} "
            f"cost/req(cold)=${ss['cost_per_req_cold_usd']}. {ss['reasoning']}"
        )
    else:
        print("\n(sweet spot not computed — need at least one cold result)")


def _f(v, nd=None):
    if v is None:
        return "NA"
    if isinstance(v, (int,)):
        return str(v)
    try:
        if nd is not None:
            return round(float(v), nd).__str__()
        return round(float(v), 2).__str__()
    except Exception:
        return str(v)


def compute_breakeven() -> dict:
    """Per-tier warm/cold break-even request rate, using plan formula and a rent model.

    Uses the true container cold start (zero-shot first request) for all
    patterns, not the per-pattern warm-start cold.
    """
    rows = _load_tier_results()
    summary_json = summarize()
    breakeven: dict = {"tiers": {}, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    for tier, tb in summary_json["tiers"].items():
        rate = tb.get("price_usd_per_hr") or 0.0
        for pat, pb in tb["patterns"].items():
            # Use the true container cold start (zero-shot) for all patterns.
            cold_s = pb.get("container_cold_start_s") or pb.get("cold_start_s")
            proc_s = pb.get("warm_n1_processing_s")
            if cold_s is None or proc_s is None:
                continue
            # Plan formula: rate at which one container is continuously busy.
            breakeven_plan = round(60.0 / (cold_s + proc_s), 2)
            # Rent-vs-cold: warm rent ($/s) breaks even against cold amortization.
            breakeven_rent = round(60.0 / cold_s, 2)
            entry = {
                "pattern": pat,
                "price_usd_per_hr": rate,
                "cold_start_s": cold_s,
                "warm_n1_processing_s": proc_s,
                "cost_per_cold_request_usd": pb.get("cost_per_req_cold_usd"),
                "warm_idle_cost_per_hour_usd": rate,
                "breakeven_req_per_min_plan": breakeven_plan,
                "breakeven_req_per_min_rent": breakeven_rent,
                "note": (
                    "Above this request rate, keeping a warm container (min_containers=1) is cheaper; "
                    "below it, scale-to-zero (cold) is cheaper. rent model = 60/cold_start; "
                    "plan model = 60/(cold+processing)."
                ),
            }
            breakeven["tiers"].setdefault(tier, {})[pat] = entry
            print(
                f"{tier}/{pat}: cold={cold_s}s proc={proc_s}s -> breakeven "
                f"{breakeven_plan} req/min (plan) / {breakeven_rent} req/min (rent)"
            )

    # Add snapshot verdict impact if present.
    svp = RESULTS / "snapshot_verdict.json"
    if svp.exists():
        try:
            breakeven["snapshot_verdict"] = json.loads(svp.read_text())
        except Exception:
            pass

    (RESULTS / "breakeven.json").write_text(json.dumps(breakeven, indent=2))
    return breakeven


# --- CLI dispatcher (local, no provisioning) -------------------------------
def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Higgs TTS Modal benchmark client / analyzer")
    sub = p.add_subparsers(dest="cmd", required=False)

    p_run = sub.add_parser("run", help="Drive an EXTERNAL server URL (no provisioning).")
    p_run.add_argument("--url", required=True)
    p_run.add_argument("--gpu-type", required=True)
    p_run.add_argument("--mode", default="warm")
    p_run.add_argument("--pattern", default="both")
    p_run.add_argument("--concurrency-levels", default="1,4,8,16")

    sub.add_parser("summarize", help="Aggregate results/*.json -> summary.json")
    sub.add_parser("breakeven", help="Compute break-even -> breakeven.json")

    p_snap = sub.add_parser("snapshot", help="Probe a DEPLOYED snapshot server for U6 verdict")
    p_snap.add_argument("--app-name", required=True)
    p_snap.add_argument("--tier", default="L4")

    args = p.parse_args(argv)
    if args.cmd is None:
        p.print_help()
        return 0
    if args.cmd == "run":
        levels = [int(x) for x in args.concurrency_levels.split(",") if x.strip()]
        files = run_benchmark(args.url, args.gpu_type, args.mode, args.pattern, levels)
        print("\n".join(files))
        return 0
    if args.cmd == "summarize":
        summarize()
        return 0
    if args.cmd == "breakeven":
        compute_breakeven()
        return 0
    if args.cmd == "snapshot":
        run_snapshot_test(args.app_name, args.tier)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

