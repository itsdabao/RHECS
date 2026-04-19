# RHECS (Work in Progress)

RHECS is a Vietnamese hallucination correction pipeline inspired by the Recursive Language Models architecture.

## Project Status
This repository is actively being updated.
Current modules are functional in prototype form, and production hardening is still in progress.

## Current Scope
- Claim extraction from Vietnamese text
- Verification flow with planner, sandbox execution, and NLI judging
- Restoration flow for surgical text replacement
- Runtime contracts and trajectory logging
- Evaluation runner and test baseline

## Main Folders
- `rhecs_core/`: Core pipeline modules (extraction, verification, restoration, runtime, logging)
- `experiments/`: PoC and evaluation scripts
- `tests/`: Runtime and pipeline tests
- `update/`: Implementation specs and execution plan
- `data/`: Input and evaluation datasets

## Notes
- The `rlm/` directory is used as a reference architecture source and is not part of this repository publish scope.
- This project is under active iteration, so interfaces and behaviors may change.
