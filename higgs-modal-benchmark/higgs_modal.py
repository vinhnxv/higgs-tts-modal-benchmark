"""Higgs TTS 3 deployment on Modal via SGLang-Omni, built for a 5-tier GPU benchmark.

The server class is parameterized by environment variables read at decoration
(import) time, so a single file deploys every GPU tier and snapshot mode:

  HIGGS_TIER       L4 | A10 | L40S | A100_40 | H100     (default L4)
  HIGGS_SNAPSHOT   0 | 1                                (default 0)
  HIGGS_APP_NAME   Modal app name                       (default higgs-tts-benchmark)

Per-tier ephemeral run (cold start + concurrency sweep):
    HIGGS_TIER=A10 modal run benchmark_client.py::benchmark --gpu-type A10

Snapshot compatibility test (requires deploy, not ephemeral run):
    HIGGS_SNAPSHOT=1 HIGGS_TIER=L4 HIGGS_APP_NAME=higgs-tts-snap-l4 modal deploy higgs_modal.py

Reference plan: docs/plans/2026-07-03-001-feat-higgs-tts-modal-benchmark-plan.md
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Optional

import modal

MINUTES = 60
PORT = 8000
MODEL_NAME = "bosonai/higgs-audio-v3-tts-4b"

GPU_TIERS: dict[str, str] = {
    "L4": "L4",
    "A10": "A10",
    "L40S": "L40S",
    "A100_40": "A100",
    "H100": "H100",
}

TIER = os.environ.get("HIGGS_TIER", "L4")
GPU = GPU_TIERS.get(TIER, "L4")
SNAPSHOT_ENABLED = os.environ.get("HIGGS_SNAPSHOT", "0") == "1"
APP_NAME = os.environ.get("HIGGS_APP_NAME", "higgs-tts-benchmark")

# SGLang-Omni tuning baseline per VRAM (KTD7). Whether the AR engine's
# `sgl-omni serve` accepts these flags is verified empirically at startup.
VRAM_GB: dict[str, int] = {"L4": 24, "A10": 24, "L40S": 48, "A100_40": 40, "H100": 80}

# Concurrency ceiling used by the sweep — kept == target/max_inputs so a single
# container absorbs the whole batch (true single-container throughput).
MAX_INPUTS = 16

ROOT = os.path.dirname(os.path.abspath(__file__))

# Volumes + secret ----------------------------------------------------------
HF_CACHE_PATH = "/data/hf_cache"
REF_PATH = "/ref_audio"
HF_CACHE_VOL = modal.Volume.from_name("higgs-hf-cache", create_if_missing=True)
REF_VOL = modal.Volume.from_name("higgs-ref-audio", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

# Image ---------------------------------------------------------------------
# Mirrors the user's existing speech/higgs-audio/server/Dockerfile: build on
# lmsysorg/sglang-omni:dev, clone sglang-omni, editable-install into a uv venv,
# put the venv on PATH so `sgl-omni serve` resolves. `add_python="3.12"` makes
# a Modal-managed Python available for the in-container helpers and the model
# downloader (KTD2; vLLM-on-Modal gist entrypoint/shm gotcha).
_image = (
    modal.Image.from_registry("lmsysorg/sglang-omni:dev", add_python="3.12")
    .entrypoint([])  # silence the image's chatty default entrypoint
    .workdir("/app")
    .run_commands(
        "git clone --depth 1 https://github.com/sgl-project/sglang-omni.git /app/sglang-omni",
        "cd /app/sglang-omni && uv venv .venv -p 3.12 && . .venv/bin/activate && uv pip install -v -e .",
    )
    .env(
        {
            "PATH": "/app/sglang-omni/.venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HF_HOME": HF_CACHE_PATH,
            "HF_HUB_CACHE": HF_CACHE_PATH,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",  # snapshot friendliness (Modal example)
        }
    )
    .run_commands(
        ". /app/sglang-omni/.venv/bin/activate && uv pip install hf-transfer==0.1.9",
    )
    .add_local_dir(os.path.join(ROOT, "reference_audio"), "/ref_audio_source")
)

app = modal.App(name=APP_NAME, image=_image)


# --- in-container HTTP helpers (stdlib only) -------------------------------
def _get(path: str, timeout: float = 5.0):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}")
    return urllib.request.urlopen(req, timeout=timeout)


def _post_json(path: str, body: dict | None = None, timeout: float = 120.0):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _check_running(p: subprocess.Popen) -> None:
    rc = p.poll()
    if rc is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def wait_ready(process: subprocess.Popen, timeout: int = 20 * MINUTES) -> None:
    """Poll /health until 200 or process dies / timeout."""
    deadline = time.time() + timeout
    last_exc: Optional[Exception] = None
    while time.time() < deadline:
        try:
            _check_running(process)  # raises CalledProcessError if subprocess died
            r = _get("/health", timeout=5)
            if 200 <= r.status < 300:
                return
        except subprocess.CalledProcessError:
            raise  # process exited (bad flag / OOM) -> surface immediately
        except Exception as e:  # not up yet
            last_exc = e
            time.sleep(3)
    raise TimeoutError(f"sgl-omni not ready within {timeout}s (last: {last_exc})")


def warmup_server() -> None:
    """Prime CUDA graphs / caches with a tiny zero-shot request."""
    body = {"input": "Hello, how are you?", "response_format": "wav"}
    for _ in range(2):
        try:
            _post_json("/v1/audio/speech", body=body, timeout=180)
            return
        except Exception:
            time.sleep(5)


def release_memory() -> None:
    """/release_memory_occupation — pre-snapshot shrinking. Raises HTTPError(404)
    if SGLang-Omni's multi-stage pipeline does not expose the endpoint."""
    _post_json("/release_memory_occupation", body={}, timeout=60)


def resume_memory() -> None:
    _post_json("/resume_memory_occupation", body={}, timeout=120)


# --- server command builder (KTD7, with empirical flag fallback) -----------
def _tuning_flags(tier: str) -> list[str]:
    vram = VRAM_GB.get(tier, 24)
    if vram <= 24:
        cuda_graph_max_bs, max_running = 4, 4
    elif vram <= 48:
        cuda_graph_max_bs, max_running = 8, 8
    else:
        cuda_graph_max_bs, max_running = 16, 16
    return ["--cuda-graph-max-bs", str(cuda_graph_max_bs), "--max-running-requests", str(max_running)]


def _full_cmd(tier: str) -> list[str]:
    return [
        "sgl-omni",
        "serve",
        "--model-path",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--allowed-local-media-path",
        REF_PATH,
    ] + _tuning_flags(tier)


def _bare_cmd() -> list[str]:
    return [
        "sgl-omni",
        "serve",
        "--model-path",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--allowed-local-media-path",
        REF_PATH,
    ]


# --- server class (U2 + U6) ------------------------------------------------
_cls_kwargs: dict = dict(
    image=_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL, REF_PATH: REF_VOL},
    secrets=[HF_SECRET],
    # Propagate tier to the container so module-level TIER reads correctly
    # (HIGGS_TIER is set locally but not in the container by default).
    env={"HIGGS_TIER": TIER, "HIGGS_SNAPSHOT": "1" if SNAPSHOT_ENABLED else "0"},
    timeout=2 * 60 * MINUTES,
    cpu=8,
    memory=8 * 1024,
    min_containers=0,
    max_containers=1,  # single-container throughput measurement
    # Required so @modal.enter(snap=True/False) methods are valid even when
    # GPU snapshotting is disabled (non-snapshot benchmark mode).
    enable_memory_snapshot=True,
)
if SNAPSHOT_ENABLED:
    _cls_kwargs["experimental_options"] = {"enable_gpu_snapshot": True}


@app.cls(**_cls_kwargs)
@modal.concurrent(target_inputs=MAX_INPUTS, max_inputs=MAX_INPUTS)
class HiggsTTS:
    """SGLang-Omni Higgs TTS server, parameterized by GPU tier and snapshot mode."""

    def __init__(self, tier: str = TIER, snapshot_mode: bool = SNAPSHOT_ENABLED) -> None:
        self.tier = tier
        self.snapshot_mode = snapshot_mode
        self.process: Optional[subprocess.Popen] = None
        self.tuning_supported = False
        self._snapshot_compatible: Optional[bool] = None

    @modal.enter(snap=True)
    def startup(self) -> None:
        """Start sgl-omni, wait for /health, warmup, and (snapshot mode) release memory."""
        attempts = [_full_cmd(self.tier), _bare_cmd()]
        last_err: Optional[Exception] = None
        for idx, cmd in enumerate(attempts):
            if self.process is not None:
                try:
                    self.process.terminate()
                except Exception:
                    pass
                time.sleep(2)
            print(f"[higgs] starting sgl-omni (attempt {idx + 1}): {' '.join(cmd)}", flush=True)
            try:
                self.process = subprocess.Popen(cmd, env=os.environ)
            except FileNotFoundError as e:
                last_err = e
                continue
            try:
                wait_ready(self.process, timeout=20 * MINUTES)
                self.tuning_supported = idx == 0
                print(
                    f"[higgs] ready (tuning_supported={self.tuning_supported}, "
                    f"snapshot_mode={self.snapshot_mode})",
                    flush=True,
                )
                break
            except Exception as e:
                last_err = e
                print(f"[higgs] start attempt {idx + 1} failed: {e}", flush=True)
                try:
                    self.process.terminate()
                except Exception:
                    pass
                self.process = None
                continue
        else:
            raise RuntimeError(f"sgl-omni serve failed to start: {last_err}")

        warmup_server()

        if self.snapshot_mode:
            try:
                release_memory()
                self._snapshot_compatible = True
                print("[higgs] snapshot: /release_memory_occupation OK (200)", flush=True)
            except urllib.error.HTTPError as e:
                self._snapshot_compatible = e.code == 200
                print(f"[higgs] snapshot: /release_memory_occupation HTTP {e.code}", flush=True)
            except Exception as e:
                self._snapshot_compatible = False
                print(f"[higgs] snapshot: /release_memory_occupation failed: {e}", flush=True)

    @modal.enter(snap=False)
    def wake_up(self) -> None:
        if self.snapshot_mode:
            try:
                resume_memory()
                print("[higgs] snapshot: /resume_memory_occupation OK", flush=True)
            except urllib.error.HTTPError as e:
                print(f"[higgs] snapshot: /resume_memory_occupation HTTP {e.code}", flush=True)
            except Exception as e:
                print(f"[higgs] snapshot: /resume_memory_occupation failed: {e}", flush=True)

    @modal.web_server(port=PORT, startup_timeout=25 * MINUTES)
    def serve(self) -> None:
        # The sgl-omni subprocess started in startup() listens on PORT; the
        # @modal.web_server proxy forwards Modal HTTP traffic to it. Nothing
        # else to do here.
        return None

    @modal.exit()
    def stop(self) -> None:
        if self.process is not None:
            try:
                self.process.terminate()
            except Exception:
                pass

    @modal.method()
    def snapshot_endpoint_status(self) -> dict:
        """Report whether the snapshot endpoints were observed compatible."""
        return {
            "tier": self.tier,
            "snapshot_mode": self.snapshot_mode,
            "compatible": self._snapshot_compatible,
            "tuning_supported": self.tuning_supported,
        }


# --- helper Modal functions (U1 + U3) --------------------------------------
@app.function(
    image=_image,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    secrets=[HF_SECRET],
    cpu=8,
    memory=4 * 1024,
    timeout=60 * 60,
)
def download_model(seed: int = 0) -> str:
    """Download the gated Higgs model into the shared HF cache Volume (one-time)."""
    import os

    from huggingface_hub import snapshot_download

    cache_repo = os.path.join(HF_CACHE_PATH, "models--" + MODEL_NAME.replace("/", "--"))
    print(f"[download] cache dir present? {os.path.isdir(cache_repo)} : {cache_repo}", flush=True)
    path = snapshot_download(repo_id=MODEL_NAME, repo_type="model", max_workers=8)
    HF_CACHE_VOL.commit()
    return f"model cached at {path}"


@app.function(
    image=_image,
    cpu=2,
    memory=2 * 1024,
    timeout=5 * MINUTES,
)
def check_shm() -> str:
    """U1 gate: log /dev/shm size and basic container resources."""
    import subprocess

    df = subprocess.run(["df", "-h", "/dev/shm"], capture_output=True, text=True).stdout
    nproc = subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip()
    mem = subprocess.run(["cat", "/proc/meminfo"], capture_output=True, text=True).stdout
    print("=== /dev/shm ===")
    print(df)
    print("=== nproc ===")
    print(nproc)
    print("=== /proc/meminfo (head) ===")
    print("\n".join(mem.splitlines()[:8]))
    return df


@app.function(
    image=_image,
    volumes={REF_PATH: REF_VOL},
    timeout=5 * MINUTES,
)
def upload_ref_audio(seed: int = 0) -> list[str]:
    """U3: stage reference audio onto the read-only Volume the server mounts at /ref_audio."""
    import os
    import shutil

    os.makedirs(REF_PATH, exist_ok=True)
    placed: list[str] = []
    for name in ["ENG_UK_M_DaveB.wav", "ENG_UK_M_DaveB.txt"]:
        src = os.path.join("/ref_audio_source", name)
        dst = os.path.join(REF_PATH, name)
        shutil.copy(src, dst)
        placed.append(dst)
        print(f"staged {src} -> {dst} ({os.path.getsize(dst)} bytes)", flush=True)
    REF_VOL.commit()
    return placed


@app.local_entrypoint()
def print_url() -> None:
    """Print the deployed server's web URL (used to drive the snapshot test)."""
    try:
        url = HiggsTTS().serve.get_web_url()
    except Exception:
        cls = modal.Cls.from_name(APP_NAME, "HiggsTTS")
        url = cls().serve.get_web_url()
    print(url)
