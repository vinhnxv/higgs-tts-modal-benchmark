# Higgs TTS Modal Benchmark — Progress

Plan: docs/plans/2026-07-03-001-feat-higgs-tts-modal-benchmark-plan.md
Branch: feat/higgs-tts-modal-benchmark
Engine: inline/subagent (only available). Strategy: serial inline (units share 2 files; cloud runs need live reaction).
Config via env at decoration time: HIGGS_TIER, HIGGS_SNAPSHOT, HIGGS_APP_NAME.

## Units
- [x] U1 Modal App scaffolding + image setup (higgs_modal.py: app, image, volumes, secret, download_model, check_shm)
- [x] U2 SGLang-Omni server class (@app.cls + @modal.web_server + enter(snap))
- [x] U3 Reference audio hosting (upload_ref_audio + --allowed-local-media-path)
- [x] U4 Benchmark client — zero-shot + voice cloning (benchmark_client.py)
- [x] U5 GPU tier sweep + cost analysis (5 tiers, results/*.json + summary)
- [x] U6 Snapshot compatibility test (snapshot_verdict.json)
- [x] U7 Warm/cold break-even analysis (breakeven.json)
- [x] Final README + cost recommendation + code review + commit

## Run plan (cost-ordered, one tier at a time)
1. Create HF Modal secret; upload ref audio (CPU).
2. U1 gates: modal build/image build, check_shm (L4), download_model (CPU, ~10GB).
3. U2/U3 verify on L4 (cheapest): health, zero-shot, voice cloning cold.
4. U5 sweep per tier in order L4 → A10 → L40S → A100_40 → H100 (zero-shot cold + voice-clone cold + warm concurrency 1/4/8/16).
5. U6 snapshot: HIGGS_SNAPSHOT=1 HIGGS_TIER=L4 deploy L4 → probe release/resume endpoints → verdict; best-effort cold reduction.
6. U7 analyze: summarize.py reads results/*.json → summary.json, breakeven.json, markdown table, recommendation.
7. Code review (ce-code-review) + commits + README.
