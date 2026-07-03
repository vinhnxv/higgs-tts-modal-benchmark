# Higgs TTS Modal Benchmark

Benchmark Higgs TTS 3 (`bosonai/higgs-audio-v3-tts-4b`) deployment on Modal across 5 GPU tiers to find the most cost-effective GPU for bursty, scale-to-zero traffic.

## Architecture

- **Server** (`higgs_modal.py`): Modal `@app.cls` deploying SGLang-Omni's `sgl-omni serve` as a subprocess, parameterized by GPU tier and snapshot mode via environment variables.
- **Benchmark client** (`benchmark_client.py`): Sends zero-shot and voice cloning requests at controlled concurrency levels, measures cold start time, throughput, latency, RTF, and calculates cost metrics.
- **Reference audio** (`reference_audio/`): WAV file + transcript for voice cloning, staged on a Modal Volume.

The server class is parameterized by environment variables read at import time:

| Variable | Default | Values |
|----------|---------|--------|
| `HIGGS_TIER` | `L4` | `L4`, `A10`, `L40S`, `A100_40`, `H100` |
| `HIGGS_SNAPSHOT` | `0` | `0` (off), `1` (on) |
| `HIGGS_APP_NAME` | `higgs-tts-benchmark` | Any Modal app name |

## Prerequisites

1. **Modal account** — authenticate with `modal token new` or `modal token set`.
2. **HuggingFace token** — the model is gated. Create a Modal secret:
   ```bash
   modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
   ```
3. **Python 3.12+** with `modal` installed (`pip install modal`).

## Running the Benchmark

### Step 1: Upload Reference Audio (one-time, CPU-only)

```bash
cd higgs-modal-benchmark
modal run higgs_modal.py::upload_ref_audio
```

### Step 2: Verify Shared Memory (one-time, CPU-only)

```bash
modal run higgs_modal.py::check_shm
```

The `/dev/shm` size is logged. If under 1 GB, the multi-stage pipeline may fail.

### Step 3: Download Model Weights (one-time, ~10 GB)

```bash
modal run higgs_modal.py::download_model
```

Downloads `bosonai/higgs-audio-v3-tts-4b` into a shared Modal Volume to avoid repeated HuggingFace downloads on each cold start.

### Step 4: Benchmark a GPU Tier (ephemeral co-run)

Each tier is benchmarked separately. The `HIGGS_TIER` env var must match `--gpu-type`:

```bash
# L4 ($0.80/hr) — cheapest viable
HIGGS_TIER=L4 modal run benchmark_client.py::benchmark --gpu-type L4

# A10 ($1.10/hr)
HIGGS_TIER=A10 modal run benchmark_client.py::benchmark --gpu-type A10

# L40S ($1.95/hr)
HIGGS_TIER=L40S modal run benchmark_client.py::benchmark --gpu-type L40S

# A100-40GB ($2.10/hr)
HIGGS_TIER=A100_40 modal run benchmark_client.py::benchmark --gpu-type A100_40

# H100 ($3.95/hr) — reference
HIGGS_TIER=H100 modal run benchmark_client.py::benchmark --gpu-type H100
```

Options:
- `--pattern zero-shot|voice-cloning|both` (default: `both`)
- `--mode cold|warm|snapshot` (default: `cold`)
- `--concurrency-levels 1,4,8,16` (default)

### Step 5: Snapshot Compatibility Test (U6)

Requires a deployed (not ephemeral) server with snapshot enabled:

```bash
# Deploy snapshot server on L4
HIGGS_SNAPSHOT=1 HIGGS_TIER=L4 HIGGS_APP_NAME=higgs-tts-snap-l4 modal deploy higgs_modal.py

# Probe snapshot endpoints and measure cold start reduction
python benchmark_client.py snapshot --app-name higgs-tts-snap-l4 --tier L4

# Tear down when done
modal app delete higgs-tts-snap-l4
```

### Step 6: Analyze Results (local, no Modal provisioning)

```bash
# Aggregate results/*.json -> summary.json + comparison table
python benchmark_client.py summarize

# Compute warm/cold break-even -> breakeven.json
python benchmark_client.py breakeven
```

## Output Structure

```
results/
├── <tier>_cold_zero-shot.json      # Cold start + concurrency sweep (zero-shot)
├── <tier>_cold_voice-cloning.json  # Cold start + concurrency sweep (voice cloning)
├── summary.json                    # Cost comparison across all tiers
├── snapshot_verdict.json           # Snapshot compatibility verdict (U6)
└── breakeven.json                  # Warm/cold break-even per tier (U7)
```

Each per-tier JSON contains:
- `cold_start.cold_start_s` — container init to first audio response
- `sweep[]` — per-concurrency-level metrics: throughput (req/s), mean latency, RTF, audio duration
- `price_usd_per_hr` — GPU rate

## GPU Tiers and Pricing

| Tier | GPU | VRAM | Price (USD/hr) |
|------|-----|------|----------------|
| L4 | L4 | 24 GB | $0.80 |
| A10 | A10 | 24 GB | $1.10 |
| L40S | L40S | 48 GB | $1.95 |
| A100_40 | A100 40GB | 40 GB | $2.10 |
| H100 | H100 | 80 GB | $3.95 |

Pricing derived from [Modal pricing](https://modal.com/pricing) per-second rates × 3600.

T4 is excluded — Turing architecture (SM 7.5) is likely incompatible with flash-attn-4, which requires Ampere+.

## Cost Methodology

- **Cost per request (cold):** `(cold_start_time + processing_time) × (gpu_rate / 3600)`
- **Cost per audio second (cold):** `(cold_start_time + processing_time) × (gpu_rate / 3600) / audio_duration`
- **Break-even request rate:** `60 / (cold_start_time + avg_processing_time)` — above this rate, keeping a warm container (`min_containers=1`) is cheaper; below it, scale-to-zero (cold) is cheaper.

The sweet spot is the tier with the lowest cost per audio second (cold-inclusive) — the primary metric for bursty, scale-to-zero traffic.

## Results

All 5 GPU tiers benchmarked with zero-shot and voice cloning patterns.

### Cost Comparison (cold-start-inclusive)

| Tier | $/hr | Pattern | Cold Start (s) | Throughput @ N=16 (req/s) | $/req (cold) | $/audio-sec (cold) |
|------|------|---------|----------------|---------------------------|--------------|---------------------|
| L4 | $0.80 | zero-shot | 290.3 | 1.78 | $0.065 | $0.034 |
| L4 | $0.80 | voice-clone | 10.1 (warm) | 0.97 | $0.003 | $0.001 |
| A10 | $1.10 | zero-shot | 237.3 | 2.74 | $0.073 | $0.043 |
| A10 | $1.10 | voice-clone | 6.9 (warm) | 1.49 | $0.003 | $0.001 |
| L40S | $1.95 | zero-shot | 274.4 | 4.66 | $0.150 | $0.089 |
| L40S | $1.95 | voice-clone | 6.9 (warm) | 3.19 | $0.005 | $0.001 |
| A100-40 | $2.10 | zero-shot | 275.5 | 5.03 | $0.162 | $0.101 |
| A100-40 | $2.10 | voice-clone | 7.5 (warm) | 3.41 | $0.006 | $0.001 |
| H100 | $3.95 | zero-shot | 197.1 | 6.44 | $0.218 | $0.136 |
| H100 | $3.95 | voice-clone | 5.4 (warm) | 5.42 | $0.008 | $0.002 |

**Note:** Voice cloning "cold start" values are warm starts — both patterns run
in the same container, so the second pattern benefits from the warm cache. The
true cold start is the zero-shot cold start (container init + model loading).

### Recommendation

For **bursty, scale-to-zero traffic** (each request triggers a cold start):

- **L4 is the most cost-effective** at $0.034/audio-second (cold-inclusive).
  The cold start time (~290s) is similar across all tiers (197-290s), so the
  cheapest GPU wins on cost per request.
- **A10 is a close second** at $0.043/audio-second, with 55% higher throughput
  (2.74 vs 1.78 req/s). If cold starts are amortized across multiple requests,
  A10 becomes competitive.
- **H100 has the shortest cold start** (197s) and highest throughput (6.44
  req/s), but at $0.136/audio-second it's 4x more expensive than L4.

For **steady-state traffic** (warm container, `min_containers=1`):

- **A10 voice cloning** is cheapest at $0.001/audio-second (warm).
- **H100 voice cloning** delivers the most audio per second at 21.7 audio-s/s.

### Break-Even Analysis

For true cold starts (zero-shot), all tiers need < 1 req/min to justify
scale-to-zero — meaning if you get less than 1 request per minute, cold starts
are cheaper than keeping a warm container.

For warm voice cloning, the break-even is 4-11 req/min depending on tier:

| Tier | Break-even (warm, req/min) |
|------|---------------------------|
| L4 | 5.9 |
| A10 | 8.7 |
| L40S | 8.7 |
| A100-40 | 8.0 |
| H100 | 11.1 |

### Snapshot Compatibility

**Verdict: Incompatible.** SGLang-Omni's multi-stage pipeline (separate
AR/codec/vocoder processes coordinated via IPC sockets) does not expose the
`/release_memory_occupation` endpoint (returns 404). Modal's GPU memory
snapshotting cannot be used to reduce cold start times for Higgs TTS.

## Reference Plan

Full plan with requirements, implementation units, and verification contract:
`docs/plans/2026-07-03-001-feat-higgs-tts-modal-benchmark-plan.md`