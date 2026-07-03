# Task for reviewer

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Review the code diff on this branch (feat/higgs-tts-modal-benchmark). This is a Modal-based GPU benchmark project for Higgs TTS 3.

Review the two main source files:
- higgs-modal-benchmark/higgs_modal.py (Modal server deployment)
- higgs-modal-benchmark/benchmark_client.py (benchmark client + analysis)

Focus on:
1. Bugs and logic errors (especially in cost calculations, concurrency handling, HTTP request logic)
2. Error handling robustness (subprocess management, timeout handling, cleanup)
3. Security (HF token handling, subprocess injection)
4. Code quality (dead code, naming, patterns)
5. Requirements coverage from the plan at docs/plans/2026-07-03-001-feat-higgs-tts-modal-benchmark-plan.md

Return findings as structured JSON with severity (P0/P1/P2/P3), file, line, title, and description. Also include an overall verdict.

## Acceptance Contract
Acceptance level: reviewed
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope
- criterion-2: Return evidence sufficient for an independent acceptance review

Required evidence: changed-files, tests-added, commands-run, validation-output, residual-risks, no-staged-files

Review gate: required by reviewer.

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```