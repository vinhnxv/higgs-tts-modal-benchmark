# Higgs TTS 3 — Modal GPU Benchmark

> Which GPU is the most cost-effective for serving Higgs TTS 3 on Modal with bursty, scale-to-zero traffic?

Benchmark of [Higgs TTS 3](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) (`bosonai/higgs-audio-v3-tts-4b`, ~4B params, BF16, 24kHz) deployed via [SGLang-Omni](https://github.com/sgl-project/sglang-omni) on [Modal](https://modal.com) across 5 GPU tiers — from the $0.80/hr L4 to the $3.95/hr H100.

## TL;DR

**Use L4 with 2 CPU cores and 2 GB memory. Total cost: ~$0.91/hr.**

| GPU | $/hr | Cold start | Throughput @N=16 | $/req (cold) | $/audio-sec (cold) |
|-----|------|------------|-------------------|--------------|---------------------|
| **L4** | **$0.80** | 290s | 1.78 req/s | **$0.065** | **$0.017** |
| A10 | $1.10 | 237s | 2.74 req/s | $0.073 | $0.019 |
| L40S | $1.95 | 274s | 4.66 req/s | $0.150 | $0.038 |
| A100-40 | $2.10 | 275s | 5.03 req/s | $0.162 | $0.039 |
| H100 | $3.95 | 197s | 6.44 req/s | $0.218 | $0.056 |

Cold start time (197–290s) is similar across all tiers because model loading (~10 GB) dominates — not GPU compute speed. So the cheapest GPU wins on cost per request.

**GPU memory snapshot: incompatible.** SGLang-Omni's multi-stage pipeline doesn't expose `/release_memory_occupation` (returns 404). Cold start reduction via Modal snapshotting is not available.

---

## What's in this repo

```
higgs-modal-benchmark/
├── higgs_modal.py          # Modal app: image, server class (@app.cls + @modal.web_server), volumes, secrets
├── benchmark_client.py     # Benchmark client: concurrency sweep, cold start, cost analysis, break-even
├── reference_audio/        # Reference audio for voice cloning (ENG_UK_M_DaveB.wav + transcript)
├── results/                # Benchmark results — 13 JSON files (5 tiers × 2 patterns + summary + breakeven + snapshot verdict)
├── README.md               # How to run the benchmark
└── REPORT.md               # Full technical report with methodology and conclusions
```

## Quick start

### Prerequisites

1. **Modal account** — `pip install modal && modal token new`
2. **HuggingFace token** — the model is gated. Create a Modal secret:
   ```bash
   modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
   ```

### Run the benchmark (one tier)

```bash
cd higgs-modal-benchmark

# 1. Upload reference audio (one-time, CPU-only)
modal run higgs_modal.py::upload_ref_audio

# 2. Download model weights (one-time, ~10 GB)
modal run higgs_modal.py::download_model

# 3. Benchmark L4 — cheapest tier
HIGGS_TIER=L4 HIGGS_CPU=2 HIGGS_MEMORY=2048 \
  modal run benchmark_client.py::benchmark --gpu-type L4 --pattern both --concurrency-levels 1,4,8,16

# 4. Analyze results (local, no GPU)
python benchmark_client.py summarize
python benchmark_client.py breakeven
```

### Recommended production deploy

```bash
HIGGS_TIER=L4 HIGGS_CPU=2 HIGGS_MEMORY=2048 modal deploy higgs_modal.py
```

All configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HIGGS_TIER` | `L4` | GPU tier: `L4`, `A10`, `L40S`, `A100_40`, `H100` |
| `HIGGS_CPU` | `8` | CPU cores (2 is sufficient) |
| `HIGGS_MEMORY` | `8192` | Memory in MB (2048 is sufficient) |
| `HIGGS_SNAPSHOT` | `0` | Enable GPU snapshot (incompatible — leave off) |

---

## Key findings

### 1. L4 is the sweet spot for bursty traffic

Cold start dominates cost for scale-to-zero traffic. Since cold start is similar across all tiers (197–290s), the cheapest GPU (L4 at $0.80/hr) delivers the lowest cost per request ($0.065) and per audio second ($0.017).

### 2. CPU and memory can be cut to 2 cores / 2 GB

The SGLang-Omni 4-stage pipeline (preprocessing → audio_encoder → tts_engine → vocoder) runs fine with 2 CPU cores and 2 GB memory. Throughput is GPU-bound — only 1–5% slower than 8 cores / 8 GB. This cuts the all-in cost from $1.24/hr to **$0.91/hr** (27% savings).

### 3. GPU memory snapshot is incompatible

SGLang-Omni uses separate processes for each pipeline stage, coordinated via IPC sockets. The `/release_memory_occupation` endpoint (required by Modal's GPU snapshot) returns 404. Cold start cannot be reduced via snapshotting.

### 4. Break-even: scale-to-zero wins below 1 req/min

For all tiers, if traffic is below ~1 request per 3–5 minutes, cold starts (scale-to-zero) are cheaper than keeping a warm container.

### 5. When to upgrade from L4

| Traffic pattern | Recommended GPU | Reason |
|-----------------|----------------|--------|
| Bursty, < 1 req/min | **L4** ($0.91/hr) | Cheapest per request |
| Moderate, 1–5 req/min | **A10** ($1.27/hr) | 55% higher throughput, only 37% more expensive |
| Continuous, high QPS | **H100** ($4.15/hr) | 6.44 req/s, shortest cold start (197s) |

---

## Full report

See [`higgs-modal-benchmark/REPORT.md`](higgs-modal-benchmark/REPORT.md) for the complete technical report: methodology, per-tier results, concurrency sweep data, snapshot verdict, CPU/memory optimization, bugs encountered, and detailed cost analysis.

---

## Tech stack

- **Model:** [Higgs TTS 3](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) — Boson AI, ~4B params, BF16, 24kHz, Vietnamese + multilingual TTS with voice cloning
- **Serving:** [SGLang-Omni](https://github.com/sgl-project/sglang-omni) — `sgl-omni serve` with multi-stage pipeline (AR engine + codec + vocoder)
- **Platform:** [Modal](https://modal.com) — serverless GPU cloud with per-second billing, scale-to-zero, and Volume persistence
- **Image:** `lmsysorg/sglang-omni:dev` — ships UCX, flash-attn-4, and SGLang prebuilt

## License

Benchmark code is provided as-is for research and cost evaluation purposes.