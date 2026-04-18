# OmniEdge E2E Harness

This suite validates mode UIs end-to-end at the browser/websocket boundary and
adds VRAM/load-time gates.

## Lanes

- `smoke`: fast regular lane, warnings allowed for unavailable browser tooling.
- `full`: strict gate (browser + ws edge checks + hard load/VRAM thresholds).
- `nightly`: strict gate plus image build.

## Entry Points

```bash
bash test.sh e2e-smoke
bash test.sh e2e-full
bash test.sh e2e-nightly
```

## Browser Requirement

Install Playwright to enable real browser action replay:

```bash
pip install -r tests/e2e/requirements.txt
playwright install chromium
```

Without Playwright:

- smoke lane records a warning.
- full/nightly fail (strict mode).

## Report

JSON output is written to:

`logs/e2e_report.json`

It includes per-mode pass/fail, stream counters, edge-case failures, load times,
and VRAM baseline/peak values.
