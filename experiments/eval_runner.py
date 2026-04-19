import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rhecs_core.extraction.extractor import extract_vietnamese_claims
from rhecs_core.verification.nli_judge import NLIStatus, judge_evidence


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preliminary evaluator for data/eval_rhecs.jsonl")
    parser.add_argument(
        "--input",
        default="data/eval_rhecs.jsonl",
        help="Path to evaluation dataset in JSONL format",
    )
    parser.add_argument(
        "--results",
        default="data/eval_results.jsonl",
        help="Path for per-sample prediction outputs (JSONL)",
    )
    parser.add_argument(
        "--summary",
        default="data/eval_summary.json",
        help="Path for summary metrics output (JSON)",
    )
    parser.add_argument(
        "--confusion",
        default="data/eval_confusion_matrix.json",
        help="Path for category confusion matrix output (JSON)",
    )
    parser.add_argument(
        "--failures",
        default="data/eval_failures.jsonl",
        help="Path for failed sample details (JSONL)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of samples to run (default: 20 for preliminary run)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of samples to skip from the start",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of samples to run concurrently",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="Per-sample timeout in seconds",
    )
    parser.add_argument(
        "--model",
        default="gemma-4-31b-it",
        help="Gemini model name to force for extractor/judge (default: gemma-4-31b-it)",
    )
    return parser.parse_args()


def configure_forced_model(model_name: str) -> None:
    # Force all relevant candidate env vars for this evaluator process.
    forced_vars = [
        "GEMINI_MODEL_CANDIDATES",
        "GOOGLE_MODEL_CANDIDATES",
        "EXTRACTOR_MODEL_CANDIDATES",
        "JUDGE_MODEL_CANDIDATES",
    ]
    for var_name in forced_vars:
        os.environ[var_name] = model_name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = " ".join(value.strip().split()).lower()
    return normalized or None


def normalize_category(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip().upper().replace(" ", "_")
    cleaned = cleaned.replace("-", "_")
    return cleaned or None


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def token_f1(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    a_tokens = a.split()
    b_tokens = b.split()
    a_counter = Counter(a_tokens)
    b_counter = Counter(b_tokens)
    overlap = sum((a_counter & b_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    return 2 * precision * recall / (precision + recall)


def model_to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def format_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


@dataclass
class SamplePrediction:
    id: str
    gold_is_hallucinated: bool
    pred_is_hallucinated: Optional[bool]
    gold_category: Optional[str]
    pred_category: Optional[str]
    gold_fault_span: Optional[str]
    pred_fault_span: Optional[str]
    verdicts: list[dict[str, Any]]
    num_claims: int
    error: Optional[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "gold_is_hallucinated": self.gold_is_hallucinated,
            "pred_is_hallucinated": self.pred_is_hallucinated,
            "gold_category": self.gold_category,
            "pred_category": self.pred_category,
            "gold_fault_span": self.gold_fault_span,
            "pred_fault_span": self.pred_fault_span,
            "num_claims": self.num_claims,
            "verdicts": self.verdicts,
            "error": self.error,
        }


async def evaluate_one(
    sample: dict[str, Any],
    timeout_sec: int,
    progress_prefix: str,
) -> SamplePrediction:
    sample_id = str(sample.get("id"))
    draft = sample.get("draft", "")
    context = sample.get("context", "")
    gold_hall = bool(sample.get("is_hallucinated", False))
    gold_category = normalize_category(sample.get("category"))
    gold_span = sample.get("expected_fault_span")

    async def _run() -> SamplePrediction:
        log(f"{progress_prefix} extracting claims...")
        claim_list = await extract_vietnamese_claims(draft)
        log(f"{progress_prefix} extracted {len(claim_list.claims)} claims")

        verdicts_raw: list[dict[str, Any]] = []
        for claim_idx, claim in enumerate(claim_list.claims, start=1):
            claim_payload = model_to_dict(claim)
            log(f"{progress_prefix} judging claim {claim_idx}/{len(claim_list.claims)}")
            verdict = await judge_evidence(draft, claim_payload, [context])
            verdicts_raw.append(model_to_dict(verdict))

        problematic = next(
            (
                v
                for v in verdicts_raw
                if v.get("status") in (NLIStatus.CONTRADICTED.value, NLIStatus.NOT_MENTIONED.value)
            ),
            None,
        )

        pred_hall = problematic is not None
        pred_category = normalize_category(problematic.get("error_category")) if problematic else None
        pred_span = problematic.get("fault_span") if problematic else None

        return SamplePrediction(
            id=sample_id,
            gold_is_hallucinated=gold_hall,
            pred_is_hallucinated=pred_hall,
            gold_category=gold_category,
            pred_category=pred_category,
            gold_fault_span=gold_span,
            pred_fault_span=pred_span,
            verdicts=verdicts_raw,
            num_claims=len(claim_list.claims),
            error=None,
        )

    try:
        return await asyncio.wait_for(_run(), timeout=timeout_sec)
    except Exception as exc:
        return SamplePrediction(
            id=sample_id,
            gold_is_hallucinated=gold_hall,
            pred_is_hallucinated=None,
            gold_category=gold_category,
            pred_category=None,
            gold_fault_span=gold_span,
            pred_fault_span=None,
            verdicts=[],
            num_claims=0,
            error=format_exception(exc),
        )


async def evaluate_all(samples: list[dict[str, Any]], concurrency: int, timeout_sec: int) -> list[SamplePrediction]:
    semaphore = asyncio.Semaphore(concurrency)
    total = len(samples)
    started_at = time.perf_counter()
    completed = 0
    ordered_results: list[Optional[SamplePrediction]] = [None] * total

    async def _worker(index: int, sample: dict[str, Any]) -> tuple[int, SamplePrediction]:
        sample_id = str(sample.get("id"))
        step_prefix = f"[Eval][{index + 1}/{total}][{sample_id}]"
        async with semaphore:
            log(f"{step_prefix} start")
            sample_started_at = time.perf_counter()
            result = await evaluate_one(sample, timeout_sec, step_prefix)
            elapsed = time.perf_counter() - sample_started_at
            status = "OK" if result.error is None else f"ERROR({result.error})"
            log(f"{step_prefix} done status={status} elapsed={elapsed:.1f}s")
            return index, result

    tasks = [asyncio.create_task(_worker(i, s)) for i, s in enumerate(samples)]
    for task in asyncio.as_completed(tasks):
        index, result = await task
        ordered_results[index] = result
        completed += 1
        elapsed_total = max(time.perf_counter() - started_at, 1e-9)
        avg_per_sample = elapsed_total / completed
        eta = avg_per_sample * (total - completed)
        log(
            f"[Eval] progress {completed}/{total} "
            f"elapsed={elapsed_total:.1f}s eta~{eta:.1f}s"
        )

    return [r for r in ordered_results if r is not None]


def compute_metrics(results: list[SamplePrediction]) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    total = len(results)
    completed = [r for r in results if r.pred_is_hallucinated is not None]
    failed = [r for r in results if r.pred_is_hallucinated is None]

    tp = sum(1 for r in completed if r.gold_is_hallucinated and r.pred_is_hallucinated)
    fp = sum(1 for r in completed if (not r.gold_is_hallucinated) and r.pred_is_hallucinated)
    fn = sum(1 for r in completed if r.gold_is_hallucinated and (not r.pred_is_hallucinated))
    tn = sum(1 for r in completed if (not r.gold_is_hallucinated) and (not r.pred_is_hallucinated))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, len(completed))

    hallucinated_gold = [r for r in completed if r.gold_is_hallucinated]
    category_scored = [r for r in hallucinated_gold if r.gold_category is not None]
    category_accuracy = safe_div(
        sum(1 for r in category_scored if r.pred_category == r.gold_category),
        len(category_scored),
    )

    span_scored = [r for r in hallucinated_gold if normalize_text(r.gold_fault_span) is not None]
    span_exact = safe_div(
        sum(
            1
            for r in span_scored
            if normalize_text(r.pred_fault_span) == normalize_text(r.gold_fault_span)
        ),
        len(span_scored),
    )
    span_overlap = safe_div(
        sum(
            token_f1(normalize_text(r.pred_fault_span), normalize_text(r.gold_fault_span))
            for r in span_scored
        ),
        len(span_scored),
    )

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in hallucinated_gold:
        expected = r.gold_category or "UNKNOWN_GOLD"
        predicted = r.pred_category or "UNKNOWN_PRED"
        confusion[expected][predicted] += 1

    confusion_json = {
        expected: dict(sorted(pred_map.items(), key=lambda kv: kv[0]))
        for expected, pred_map in sorted(confusion.items(), key=lambda kv: kv[0])
    }

    summary = {
        "sample_counts": {
            "total": total,
            "completed": len(completed),
            "failed": len(failed),
            "coverage": safe_div(len(completed), total),
        },
        "detection": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        },
        "localization": {
            "exact_match": span_exact,
            "token_overlap_f1": span_overlap,
            "scored_samples": len(span_scored),
        },
        "category": {
            "accuracy": category_accuracy,
            "scored_samples": len(category_scored),
        },
        "notes": {
            "mode": "preliminary_extractor_plus_judge",
            "prediction_rule": "hallucinated if any verdict is CONTRADICTED or NOT_MENTIONED",
        },
    }

    return summary, confusion_json


def main() -> None:
    load_dotenv()
    args = parse_args()
    configure_forced_model(args.model)

    input_path = Path(args.input)
    rows = read_jsonl(input_path)
    rows = rows[args.offset :]
    if args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        raise ValueError("No samples selected. Adjust --offset/--limit or input file.")

    log(f"[Eval] Forced model: {args.model}")
    log(f"[Eval] Loaded {len(rows)} samples from {input_path}")
    log(f"[Eval] Running with concurrency={args.concurrency}, timeout={args.timeout_sec}s")

    results = asyncio.run(evaluate_all(rows, args.concurrency, args.timeout_sec))

    result_rows = [r.to_json() for r in results]
    write_jsonl(Path(args.results), result_rows)

    failures = [r.to_json() for r in results if r.error is not None]
    write_jsonl(Path(args.failures), failures)

    summary, confusion = compute_metrics(results)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.summary).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with Path(args.confusion).open("w", encoding="utf-8") as f:
        json.dump(confusion, f, ensure_ascii=False, indent=2)

    log("[Eval] Completed.")
    log(f"[Eval] Results: {args.results}")
    log(f"[Eval] Summary: {args.summary}")
    log(f"[Eval] Confusion: {args.confusion}")
    log(f"[Eval] Failures: {args.failures}")


if __name__ == "__main__":
    main()
