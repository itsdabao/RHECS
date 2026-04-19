# 06 Evaluation Runner

## Why
Without a standard eval runner, quality improvements are anecdotal and hard to compare across iterations.

## Inputs
Primary dataset:
- `data/eval_rhecs.jsonl`

Each record should include:
- id
- input_text
- expected_fault_span
- expected_category
- expected_correction (optional)

## Outputs
- `data/eval_results.jsonl`
- `data/eval_summary.json`
- `data/eval_confusion_matrix.json`
- `data/eval_failures.jsonl`

## Metrics
- Detection precision/recall/F1
- Localization exact match
- Localization overlap score
- Category accuracy
- Per-category precision/recall/F1
- Failure rate by stage

## Runner Behavior
- Resume-safe by sample id
- Configurable concurrency
- Per-sample timeout
- Retry only for retryable classes
- Persist partial progress every N samples

## Repository Changes
- Add `experiments/eval_runner.py`
- Add `experiments/metrics.py`
- Add `experiments/reporting.py`

## Implementation Steps
1. Load dataset and normalize records.
2. Run pipeline for each sample with controlled concurrency.
3. Store raw prediction + runtime metadata per sample.
4. Compute metrics and confusion matrix.
5. Emit summary and failure breakdown.

## Definition of Done
- Full dataset run can be resumed after interruption.
- Summary file includes global and category metrics.
- Failure file includes trace references for debugging.

## Validation
- Small 20-sample dry run
- Full run reproducibility check
- Spot-check misclassified samples
