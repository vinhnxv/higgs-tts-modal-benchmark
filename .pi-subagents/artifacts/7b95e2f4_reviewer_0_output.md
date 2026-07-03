## Review Summary — `2026-07-03-001-feat-higgs-tts-modal-benchmark-plan.md`

**Artifact location note (FYI):** The task path `/Users/vinhnxv/docs/plans/...` does not exist. The actual artifact is at `/Users/vinhnxv/Desktop/repos/vinhnxv/HiggsAudioTTSModal/docs/plans/2026-07-03-001-feat-higgs-tts-modal-benchmark-plan.md`. Not a defect of the plan itself.

### Requirements traceability — PASS (with one intra-requirement defect)
All R1–R15 are covered by ≥1 implementation unit:
- R1,R2,R3,R4 → U1; R1,R6 → U2; R5,R7,R8,R9,R10 → U4/U5; R11 → U6; R12 → U7; R13,R14,R15 → U3/U4. No orphan requirements; no orphan units.

### Findings by severity

**P1 — Important (must resolve before implementation)**

1. **Wrong model repo ID throughout** *(auto-fixed)* — Plan used `bosonai/higgs-tts-3-4b` (lines 16, 52, 109, 226, 254). The user's working setup (`speech/higgs-audio/server/docker-compose.yml`, `client/higgs_tts_vi.py`) and a local copy of the model card (`repos/higgs_audio/higgs-audio-v3-tts-4b.md`) confirm the canonical ID is **`bosonai/higgs-audio-v3-tts-4b`**. As written, `hf download` and `sgl-omni serve --model-path` would target a non-existent/wrong repo. → Corrected to `bosonai/higgs-audio-v3-tts-4b` in all 5 locations. (Borders on P0; downgraded to P1 because it is a single-string fix and the architecture specs in the plan already match the real card.)

2. **A100-40GB pricing contradiction** *(not auto-fixed — true value unverifiable offline)* — R5 (line 56) lists `A100-40GB ($2.10/hr)`; U5 pricing constants (line 344) define `A100=1.95`. These disagree and feed directly into the cost recommendation. Requires manual reconciliation against current Modal pricing.

3. **Reference audio path/filename/format wrong** *(auto-fixed)* — Plan cited `speech/higgs-audio/voices/` and `male-voice.wav`/`ENG_UK_M_DaveB.wav`. Verified: reference audio lives in **`speech/voices/ENG_UK_M_DaveB.mp3`** (MP3, not WAV); the `speech/higgs-audio/voices/` dir does not exist. The user's `xiaomimimo/higgs_audio.py` converts MP3→WAV via `ffmpeg`. Plan was missing the ffmpeg conversion step and ffmpeg-in-image dependency. → Fixed path to `speech/voices/`, filename to `ENG_UK_M_DaveB.wav` (converted from `.mp3`), and added an ffmpeg availability check/conversion note in U3.

4. **Reference transcript mismatch** *(auto-fixed)* — Plan used a fabricated transcript "Hey, Adam here..." for the voice-cloning reference. Voice cloning requires the reference `text` to match the spoken content of the reference audio; the real transcript (verified from `speech/voices/ENG_UK_M_DaveB.txt`) is the "Sodi Scientifica…" text. A mismatched transcript degrades clone validity and would corrupt the benchmark. → Replaced with the real transcript and a pointer to the full transcript file.

5. **Undefined "minimum throughput threshold" for the sweet-spot recommendation** *(not auto-fixable — product decision)* — U5 says "the tier with the lowest cost per request that still meets a minimum throughput threshold," but that threshold is never defined anywhere. The Success Criterion "Clear recommendation" is therefore subjective. Needs an explicit threshold (e.g., a target req/s or RTF ceiling) before the recommendation can be attested.

6. **Missing brainstorm source artifact** — Frontmatter declares `product_contract_source: ce-brainstorm` and the plan states "Product Contract unchanged — all R-IDs … preserved from the brainstorm," but no brainstorm artifact exists in the repo (search returned none). External traceability for the "unchanged from brainstorm" claim is broken. (The Product Contract is embedded, so internal scope alignment is still checkable — see below.)

**FYI — Informational**

7. **Scope alignment — PASS.** Implementation Units map cleanly to the three benchmark approaches (A→U4/U5, B→U6, C→U7) and respect all scope boundaries (streaming, inline-control, multi-GPU TP, vLLM, production hardening all deferred). Embedded Product Contract (Summary/Problem Frame/Requirements/Scope/Success Criteria) is internally consistent with the units.

8. **Stop-condition tension (FYI).** Goal Capsule says "for all 5 GPU tiers" while DoD allows "failures documented for incompatible tiers." The DoD's "valid tiers" caveat is the operative one; consider softening the stop-condition wording to "all viable tiers" for consistency.

9. **RTF definition ambiguity (FYI).** U4 defines RTF = processing_time / audio_duration but "processing_time" is ambiguous at concurrency >1 (latency includes queueing). Recommend specifying that RTF is computed at concurrency=1, or define server-side processing time distinctly from client-observed latency.

10. **Unverified Modal SDK API claims (FYI).** KTD1's assertion that `@modal.web_server` supports `enable_memory_snapshot` while `@app.server` does not; `add_python="3.12"`; `@modal.concurrent(target_inputs=…, max_inputs=…)`; `experimental_options={"enable_gpu_snapshot": True}`. Modal's SDK churns — these names/policy should be confirmed against the current SDK before implementation. (The plan does acknowledge snapshot support as empirically resolved in U6, which mitigates the highest-risk one.)

11. **Break-even formula is a single-container model (FYI).** U7's `break_even_req_per_min = 60 / (cold_start + avg_processing)` is correct for the single-container scale-to-zero vs. `min_containers=1` comparison, but it ignores multi-container autoscaling capacity (a warm container saturates at `1/processing` req/s). Acceptable for this scope (single-GPU, scale-to-zero) but should be stated explicitly as a modeling assumption.

12. **Implementation detail in product-scope requirements (FYI).** R1 (`sgl-omni serve` command), R2 (exact Docker image), R14 (Modal hosting mechanism) couple product scope to a specific implementation. This is consistent with the `ce-unified-plan/v1` artifact style (Product Contract "enriched with implementation detail below"), but it reduces implementation optionality. Acceptable as-is.

13. **Benchmark `input` language mismatch (FYI).** Objective emphasizes Vietnamese TTS, but the benchmark `input` samples are English ("Have a nice day…"). Cost is largely length-driven so this is fine for a cost benchmark, but adding a Vietnamese input sample would better match the stated use case.

### Safe auto-fixes applied
Applied only high-confidence corrections verified against the user's actual repo artifacts:
- Model ID `bosonai/higgs-tts-3-4b` → `bosonai/higgs-audio-v3-tts-4b` (5 locations).
- Reference audio source `speech/higgs-audio/voices/` → `speech/voices/`.
- Reference audio filename `male-voice.wav` → `ENG_UK_M_DaveB.wav` (+ MP3→WAV conversion note + ffmpeg-in-image check).
- Reference transcript "Hey, Adam here…" → real "Sodi Scientifica…" transcript (+ pointer to `ENG_UK_M_DaveB.txt`).

Left for manual resolution (not safe to auto-fix offline): the A100 pricing contradiction (P1-2), the undefined throughput threshold (P1-5), and the brainstorm-source traceability gap (P1-6).