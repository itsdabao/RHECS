# RHECS — RAG Hallucination Evidence-based Correction System

A production-hardened Vietnamese hallucination correction pipeline inspired by the Recursive Language Models (RLM) architecture. RHECS detects factual errors in LLM-generated Vietnamese text by extracting atomic claims, verifying them against a Qdrant vector database, and surgically patching only the incorrect spans — preserving the author's original voice.

## Architecture

```
Input Text
    │
    ▼
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Extraction  │────▶│   Verification   │────▶│   Restoration   │
│  (Claims)    │     │  (Judge + Router) │     │  (Patch)        │
└──────────────┘     └──────────────────┘     └─────────────────┘
                            │
                     ┌──────┴──────┐
                     │  Sandbox    │
                     │  (Policy    │
                     │   Guard)    │
                     └─────────────┘
```

### Core Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| **Extraction** | `rhecs_core/extraction/` | Decomposes text into atomic `(entity, relationship, target)` claims with `ClaimStatus` tracking |
| **Verification** | `rhecs_core/verification/` | AST-secured sandbox execution + NLI Judge with evidence-grounded reasoning |
| **Restoration** | `rhecs_core/restoration/` | Multi-query evidence compilation + 3-tier surgical text replacement |
| **Query Strategy** | `rhecs_core/query_strategy/` | Strategy router supporting `direct_llm` and `rlm_recursive` verification modes |
| **Runtime** | `rhecs_core/runtime/` | State machine, contracts, centralized config (`strategy`, `budget`, `timeout`) |

## Key Features

- **AST Policy Guard** — Blocks dangerous code (`subprocess`, `eval`, `exec`, `os.system`) before sandbox execution
- **Structured Error Taxonomy** — 7 error types with intelligent retry policy (non-retryable errors skip retry immediately)
- **Unresolved Ambiguity Tracking** — Claims with pronoun ambiguity are flagged `unresolved_ambiguity` instead of being dropped
- **Multi-Query Evidence Compiler** — Triplet → entity-only → metadata-enriched query expansion with deduplication and provenance
- **3-Tier Surgical Replacer** — Exact → Normalized → Fuzzy matching with configurable thresholds
- **Centralized Agent Prompts** — All 4 LLM agent prompts managed in `rhecs_core/prompts.py` with quality checklist validation
- **Strategy A/B Evaluation** — `--strategy` flag for benchmarking `direct_llm` vs `rlm_recursive`

## Project Structure

```
rhecs_core/               # Core pipeline
├── extraction/           #   Claim extraction + ClaimStatus enum
├── verification/         #   Root planner, NLI judge, sandbox, policy guard
├── restoration/          #   Evidence compiler, rewriter, surgical replacer
├── query_strategy/       #   QueryRouter + RLM bridge + DirectLLM adapter
├── runtime/              #   State machine, contracts, config
├── llm/                  #   Model router with fallback
├── logging/              #   Trajectory logging
├── prompts.py            #   Centralized prompt templates (4 agents)
└── pipeline.py           #   Main orchestrator

tests/                    # 164 tests (unit + regression golden pack)
experiments/              # Evaluation runner + simulation scripts
scripts/                  # Data preparation utilities
docs/architecture/        # Architecture specification documents
data/                     # Datasets and evaluation data
```

## Quick Start

### Prerequisites
- Python 3.11+
- A Gemini API key

### Installation

```bash
# Clone and install dependencies
git clone <repo-url> && cd RHECS
pip install -r requirements.txt  # or: pip install pydantic google-genai tenacity python-dotenv qdrant-client

# Set up environment
cp .env.example .env
# Edit .env with your GEMINI_API_KEY
```

### Run Tests

```bash
# Full regression suite (164 tests, no LLM calls needed)
python -m pytest tests/ -v

# Run only the golden regression pack
python -m pytest tests/test_regression_pack.py -v
```

### Run Evaluation

```bash
# Direct LLM strategy
python experiments/eval_runner.py --strategy direct_llm

# RLM Recursive strategy
python experiments/eval_runner.py --strategy rlm_recursive
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `RHECS_VERIFICATION_STRATEGY` | `direct_llm` | Default verification strategy |
| `RHECS_ROUTER_MAX_RECURSION` | `2` | Max recursion depth for RLM |
| `RHECS_ROUTER_MAX_CALLS` | `10` | Max total LLM calls per verification |
| `RHECS_ROUTER_TIMEOUT_MS` | `30000` | Max total time budget (ms) |
| `RHECS_EVIDENCE_TOP_K` | `3` | Top-K evidence retrieved per query |
| `RHECS_FUZZY_THRESHOLD` | `0.8` | Fuzzy matching sensitivity for surgical replacer |

## Test Coverage

| Test File | Tests | Coverage Area |
|-----------|-------|---------------|
| `test_agent_prompts.py` | 29 | Prompt templates + quality checklist |
| `test_extraction_quality.py` | 13 | ClaimStatus, metadata validation |
| `test_policy_guard.py` | 29 | AST security (imports, calls, attributes) |
| `test_sandbox_errors.py` | 25 | Error taxonomy, retry policy, telemetry |
| `test_regression_pack.py` | 27 | Golden tests (contradiction, restoration, safety) |
| `test_restoration_quality.py` | 22 | Evidence compiler, surgical replacer |
| `test_runtime_*.py` | 13 | State machine, config, contracts |
| `test_query_router_*.py` | 6 | Direct LLM adapter, RLM bridge, degradation |
| **Total** | **164** | **Full pipeline coverage** |

## License

This project is developed for academic research purposes (CS431 course).
