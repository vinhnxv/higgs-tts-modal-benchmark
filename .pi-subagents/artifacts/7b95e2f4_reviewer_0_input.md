# Task for reviewer

Review the implementation-ready plan at /Users/vinhnxv/docs/plans/2026-07-03-001-feat-higgs-tts-modal-benchmark-plan.md in headless mode. Check for: (1) gaps in requirements traceability (are all R-IDs covered by implementation units?), (2) contradictions between sections, (3) weak premises or unsupported assumptions, (4) scope alignment issues (does the plan match the Product Contract?), (5) missing test scenarios or verification gaps, (6) implementation detail leaking into what should be product scope, (7) any structural issues with the plan artifact. Return a structured summary of findings with severity levels (P0 critical, P1 important, FYI informational) and any safe auto-fixes that should be applied.

## Acceptance Contract
Acceptance level: attested
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Return concrete findings with file paths and severity when applicable

Required evidence: review-findings, residual-risks

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