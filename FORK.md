# Fork notes

**Upstream:** https://github.com/kyegomez/open-jev @ `93843ef` (2026-09-20), Apache-2.0.
**Purpose:** a self-hosted, sovereign typed-decision model (Noul / Choice / Score
with calibrated probabilities and separate epistemic confidence). No third-party
inference API in the loop.

## v0.1.0: upstream defects fixed

Each fix is pinned by a test in `tests/test_invariants.py`.

| # | Upstream behaviour | Fix |
|---|---|---|
| F1 | Claimed key-order invariance; shuffled dict keys changed sibling indices and flat positions, so answers drifted (0.730 → 0.707 in a probe). | Dict keys are visited in sorted order. Shuffled keys give identical tokens, paths and answers. List order is still preserved. |
| F3 | Confidence loss detached only its target; its gradient still flowed into the shared state encoder and read-out stack. | The confidence head reads a detached read-out vector. Test asserts no gradient reaches any other parameter. |
| F4 | `max_options=255` documented, not enforced (1,000 options accepted); oversized `Score` rubrics failed deep in the head. | `Jev.validate()` raises a clear `ValueError` for both caps. |
| F5 | States past `max_state_len` truncated silently. | `UserWarning` plus a per-state `StateCache.truncated` flag. |
| F6 | `StateCache` existed but `forward()` always re-encoded; the state was copied once per question (`[B·N, T, d]`). | Public `encode_state()` → `ask(cache, questions)`. All questions share one `[B, N·M, d]` slot sequence with a block-diagonal mask, so no per-question state copy. Question independence still holds exactly (tested). |
| F7 | `Noul` returned no confidence, so binary questions couldn't trigger escalation. | `NoulAnswer.confidence`. |
| — | No packaging or tests; README pointed at a missing `docs/ARCHITECTURE.md` and a demo that isn't in `main.py`. | `pyproject.toml`, pytest suite, README corrected. |

Not fixed here (F2): the hash tokenizer (heavy collisions at `vocab_size=4096`).
It gets replaced along with the from-scratch encoder in the next item.

## Queue

1. **v0.1.0: defect fixes** ← this release
2. Pretrained bidirectional encoder (ModernBERT / DeBERTa-v3 class) plus its
   tokenizer as the `StateEncoder` / `TextEncoder` backbone
3. Teacher pipeline: open-weight LLMs on on-prem inference, sampled across
   paraphrases and option orders, producing soft targets for `RLCDLoss`
4. Baseline harness: local LLM with grammar-constrained JSON, and NLI zero-shot
5. Calibration gate: ECE and reliability on held-out plus shifted data before
   any unattended use

## Upstream sync

```bash
git remote add upstream https://github.com/kyegomez/open-jev
git fetch upstream && git log --oneline main..upstream/main
```
