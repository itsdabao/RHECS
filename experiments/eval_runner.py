import argparse
import asyncio
import json
import os
import sys
import threading
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


@dataclass
class EvalRuntimeConfig:
    evidence_source: str
    fallback_to_context: bool
    strategy: str = "direct_llm"  # WS5-01: verification strategy


class QdrantEvidenceRetriever:
    def __init__(
        self,
        qdrant_url: str,
        collection: str,
        tenant_id: str,
        embedding_model: str,
        top_k: int,
    ) -> None:
        self.qdrant_url = qdrant_url
        self.collection = collection
        self.tenant_id = tenant_id
        self.embedding_model = embedding_model
        self.top_k = top_k
        self._client = None
        self._model = None
        self._lock = threading.Lock()

    def _ensure_ready(self) -> None:
        if self._client is not None and self._model is not None:
            return
        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant retrieval mode requires qdrant-client and sentence-transformers. "
                "Install with: D:/miniconda3/envs/rhecs/python.exe -m pip install qdrant-client sentence-transformers"
            ) from exc

        self._client = QdrantClient(url=self.qdrant_url)
        self._model = SentenceTransformer(self.embedding_model)

    @staticmethod
    def _build_query(claim_payload: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in ("entity", "relationship", "target"):
            value = claim_payload.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    parts.append(text)

        metadata = claim_payload.get("metadata")
        if isinstance(metadata, dict):
            for key in ("time", "location", "condition"):
                value = metadata.get(key)
                if value is not None:
                    text = str(value).strip()
                    if text:
                        parts.append(text)

        query = " ".join(parts).strip()
        if query:
            return query
        return json.dumps(claim_payload, ensure_ascii=False)

    def retrieve(self, claim_payload: dict[str, Any]) -> list[str]:
        self._ensure_ready()
        if self._client is None or self._model is None:
            raise RuntimeError("Retriever initialization failed")

        query_text = self._build_query(claim_payload)

        with self._lock:
            query_vector = self._model.encode(
                [f"query: {query_text}"],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0].tolist()

            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=self.tenant_id),
                    )
                ]
            )

            if hasattr(self._client, "search"):
                results = self._client.search(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=self.top_k,
                )
            else:
                query_response = self._client.query_points(
                    collection_name=self.collection,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=self.top_k,
                )
                results = getattr(query_response, "points", query_response)

        evidences: list[str] = []
        for hit in results:
            payload = hit.payload if hasattr(hit, "payload") else hit.get("payload", {})
            text = payload.get("text") if isinstance(payload, dict) else None
            if text:
                evidences.append(str(text))

        return evidences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preliminary evaluator for data/eval_rhecs.jsonl"
    )
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
    parser.add_argument(
        "--evidence-source",
        choices=["qdrant", "context"],
        default="qdrant",
        help="Evidence source for judge: qdrant retrieval or direct context (default: qdrant)",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL used when evidence-source=qdrant",
    )
    parser.add_argument(
        "--qdrant-collection",
        default=os.getenv("QDRANT_COLLECTION", "truth_base"),
        help="Qdrant collection used when evidence-source=qdrant",
    )
    parser.add_argument(
        "--tenant-id",
        default=os.getenv("TENANT_ID", "viquad"),
        help="Tenant id filter for Qdrant retrieval",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=3,
        help="Top-K evidence chunks fetched per claim when using qdrant",
    )
    parser.add_argument(
        "--retrieval-embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"),
        help="Embedding model used to encode claim query for Qdrant search",
    )
    parser.add_argument(
        "--fallback-to-context",
        action="store_true",
        help="If retrieval returns empty, fallback to sample context as judge evidence",
    )
    parser.add_argument(
        "--strategy",
        choices=["direct_llm", "rlm_recursive"],
        default="direct_llm",
        help="Verification strategy: direct_llm (default) or rlm_recursive",
    )
    parser.add_argument(
        "--sweep-top-k",
        default="",
        help="Comma-separated top-k values for retrieval sweep (example: 1,3)",
    )
    parser.add_argument(
        "--sweep-report",
        default="data/eval_topk_sweep.json",
        help="Path for sweep comparison report (JSON)",
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


def parse_top_k_sweep(raw: str) -> list[int]:
    if not raw.strip():
        return []

    output: list[int] = []
    seen: set[int] = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid top-k value '{token}' in --sweep-top-k") from exc
        if value <= 0:
            raise ValueError("All top-k values in --sweep-top-k must be > 0")
        if value not in seen:
            seen.add(value)
            output.append(value)

    if not output:
        raise ValueError(
            "--sweep-top-k was provided but no valid top-k values were parsed"
        )
    return output


def with_token_suffix(path: Path, token: str) -> Path:
    return path.with_name(f"{path.stem}_{token}{path.suffix}")


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


def _percentile(values: list, pct: float) -> float:
    """Compute percentile from a list of numeric values."""
    filtered = sorted(v for v in values if v is not None)
    if not filtered:
        return 0.0
    idx = int(len(filtered) * pct)
    idx = min(idx, len(filtered) - 1)
    return float(filtered[idx])


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
    claims_without_evidence: int
    error: Optional[str]
    latency_ms: Optional[int] = None  # WS5-01: per-sample timing
    strategy_used: Optional[str] = None  # WS5-01: strategy tracking

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
            "claims_without_evidence": self.claims_without_evidence,
            "verdicts": self.verdicts,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "strategy_used": self.strategy_used,
        }


async def evaluate_one(
    sample: dict[str, Any],
    timeout_sec: int,
    progress_prefix: str,
    runtime_cfg: EvalRuntimeConfig,
    retriever: Optional[QdrantEvidenceRetriever],
) -> SamplePrediction:
    sample_id = str(sample.get("id"))
    draft = sample.get("draft", "")
    context = sample.get("context", "")
    gold_hall = bool(sample.get("is_hallucinated", False))
    gold_category = normalize_category(sample.get("category"))
    gold_span = sample.get("expected_fault_span")

    async def _run() -> SamplePrediction:
        sample_start = time.perf_counter()
        log(f"{progress_prefix} extracting claims... [strategy={runtime_cfg.strategy}]")
        claim_list = await extract_vietnamese_claims(draft)
        log(f"{progress_prefix} extracted {len(claim_list.claims)} claims")

        verdicts_raw: list[dict[str, Any]] = []
        claims_without_evidence = 0
        for claim_idx, claim in enumerate(claim_list.claims, start=1):
            claim_payload = model_to_dict(claim)

            evidence_source = runtime_cfg.evidence_source
            evidence_list: list[str]
            if runtime_cfg.evidence_source == "qdrant":
                if retriever is None:
                    raise RuntimeError(
                        "Evidence source is qdrant but retriever is not configured"
                    )
                evidence_list = await asyncio.to_thread(
                    retriever.retrieve, claim_payload
                )
                if not evidence_list:
                    claims_without_evidence += 1
                    if runtime_cfg.fallback_to_context and context:
                        evidence_list = [context]
                        evidence_source = "qdrant_fallback_context"
            else:
                evidence_list = [context] if context else []

            log(
                f"{progress_prefix} judging claim {claim_idx}/{len(claim_list.claims)} "
                f"evidence={len(evidence_list)} source={evidence_source}"
            )
            verdict = await judge_evidence(draft, claim_payload, evidence_list)
            verdict_json = model_to_dict(verdict)
            verdict_json["_evidence_count"] = len(evidence_list)
            verdict_json["_evidence_source"] = evidence_source
            verdicts_raw.append(verdict_json)

        problematic = next(
            (
                v
                for v in verdicts_raw
                if v.get("status")
                in (NLIStatus.CONTRADICTED.value, NLIStatus.NOT_MENTIONED.value)
            ),
            None,
        )

        pred_hall = problematic is not None
        pred_category = (
            normalize_category(problematic.get("error_category"))
            if problematic
            else None
        )
        pred_span = problematic.get("fault_span") if problematic else None

        sample_elapsed_ms = int((time.perf_counter() - sample_start) * 1000)
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
            claims_without_evidence=claims_without_evidence,
            error=None,
            latency_ms=sample_elapsed_ms,
            strategy_used=runtime_cfg.strategy,
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
            claims_without_evidence=0,
            error=format_exception(exc),
        )


async def evaluate_all(
    samples: list[dict[str, Any]],
    concurrency: int,
    timeout_sec: int,
    runtime_cfg: EvalRuntimeConfig,
    retriever: Optional[QdrantEvidenceRetriever],
) -> list[SamplePrediction]:
    semaphore = asyncio.Semaphore(concurrency)
    total = len(samples)
    started_at = time.perf_counter()
    completed = 0
    ordered_results: list[Optional[SamplePrediction]] = [None] * total

    async def _worker(
        index: int, sample: dict[str, Any]
    ) -> tuple[int, SamplePrediction]:
        sample_id = str(sample.get("id"))
        step_prefix = f"[Eval][{index + 1}/{total}][{sample_id}]"
        async with semaphore:
            log(f"{step_prefix} start")
            sample_started_at = time.perf_counter()
            result = await evaluate_one(
                sample,
                timeout_sec,
                step_prefix,
                runtime_cfg=runtime_cfg,
                retriever=retriever,
            )
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


def compute_metrics(
    results: list[SamplePrediction],
    runtime_cfg: EvalRuntimeConfig,
    retrieval_top_k: int,
    tenant_id: str,
    qdrant_collection: str,
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    total = len(results)
    completed = [r for r in results if r.pred_is_hallucinated is not None]
    failed = [r for r in results if r.pred_is_hallucinated is None]

    tp = sum(1 for r in completed if r.gold_is_hallucinated and r.pred_is_hallucinated)
    fp = sum(
        1 for r in completed if (not r.gold_is_hallucinated) and r.pred_is_hallucinated
    )
    fn = sum(
        1 for r in completed if r.gold_is_hallucinated and (not r.pred_is_hallucinated)
    )
    tn = sum(
        1
        for r in completed
        if (not r.gold_is_hallucinated) and (not r.pred_is_hallucinated)
    )

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

    span_scored = [
        r for r in hallucinated_gold if normalize_text(r.gold_fault_span) is not None
    ]
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
            token_f1(
                normalize_text(r.pred_fault_span), normalize_text(r.gold_fault_span)
            )
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
            "claims_without_evidence": sum(
                r.claims_without_evidence for r in completed
            ),
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
        "latency": {
            "mean_ms": safe_div(
                sum(r.latency_ms for r in completed if r.latency_ms is not None),
                sum(1 for r in completed if r.latency_ms is not None),
            ),
            "p50_ms": _percentile(
                [r.latency_ms for r in completed if r.latency_ms is not None], 0.5
            ),
            "p95_ms": _percentile(
                [r.latency_ms for r in completed if r.latency_ms is not None], 0.95
            ),
            "max_ms": max(
                (r.latency_ms for r in completed if r.latency_ms is not None), default=0
            ),
        },
        "notes": {
            "mode": (
                "extractor_plus_judge_qdrant_retrieval"
                if runtime_cfg.evidence_source == "qdrant"
                else "extractor_plus_judge_context"
            ),
            "prediction_rule": "hallucinated if any verdict is CONTRADICTED or NOT_MENTIONED",
            "strategy": runtime_cfg.strategy,
            "evidence_source": runtime_cfg.evidence_source,
            "fallback_to_context": runtime_cfg.fallback_to_context,
            "retrieval_top_k": retrieval_top_k,
            "tenant_id": tenant_id,
            "qdrant_collection": qdrant_collection,
        },
    }

    return summary, confusion_json


def run_single_eval(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    runtime_cfg: EvalRuntimeConfig,
    retrieval_top_k: int,
    results_path: Path,
    summary_path: Path,
    confusion_path: Path,
    failures_path: Path,
) -> tuple[dict[str, Any], float]:
    retriever: Optional[QdrantEvidenceRetriever] = None
    if runtime_cfg.evidence_source == "qdrant":
        retriever = QdrantEvidenceRetriever(
            qdrant_url=args.qdrant_url,
            collection=args.qdrant_collection,
            tenant_id=args.tenant_id,
            embedding_model=args.retrieval_embedding_model,
            top_k=retrieval_top_k,
        )

    started_at = time.perf_counter()
    results = asyncio.run(
        evaluate_all(
            rows,
            args.concurrency,
            args.timeout_sec,
            runtime_cfg=runtime_cfg,
            retriever=retriever,
        )
    )
    elapsed = time.perf_counter() - started_at

    result_rows = [r.to_json() for r in results]
    write_jsonl(results_path, result_rows)

    failures = [r.to_json() for r in results if r.error is not None]
    write_jsonl(failures_path, failures)

    summary, confusion = compute_metrics(
        results,
        runtime_cfg=runtime_cfg,
        retrieval_top_k=retrieval_top_k,
        tenant_id=args.tenant_id,
        qdrant_collection=args.qdrant_collection,
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    with confusion_path.open("w", encoding="utf-8") as f:
        json.dump(confusion, f, ensure_ascii=False, indent=2)

    return summary, elapsed


def main() -> None:
    load_dotenv()
    args = parse_args()
    configure_forced_model(args.model)

    if args.retrieval_top_k <= 0:
        raise ValueError("--retrieval-top-k must be > 0")

    sweep_top_k_values = parse_top_k_sweep(args.sweep_top_k)

    input_path = Path(args.input)
    rows = read_jsonl(input_path)
    rows = rows[args.offset :]
    if args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        raise ValueError("No samples selected. Adjust --offset/--limit or input file.")

    runtime_cfg = EvalRuntimeConfig(
        evidence_source=args.evidence_source,
        fallback_to_context=args.fallback_to_context,
        strategy=args.strategy,
    )

    if sweep_top_k_values and runtime_cfg.evidence_source != "qdrant":
        raise ValueError(
            "--sweep-top-k is only supported when --evidence-source=qdrant"
        )

    log(f"[Eval] Forced model: {args.model}")
    log(f"[Eval] Strategy: {runtime_cfg.strategy}")
    log(f"[Eval] Loaded {len(rows)} samples from {input_path}")
    log(f"[Eval] Evidence source: {runtime_cfg.evidence_source}")
    if runtime_cfg.evidence_source == "qdrant":
        log(
            f"[Eval] Retrieval config: url={args.qdrant_url} "
            f"collection={args.qdrant_collection} tenant={args.tenant_id} top_k={args.retrieval_top_k}"
        )
        if runtime_cfg.fallback_to_context:
            log(
                "[Eval] Retrieval fallback: enabled (context used when no evidence found)"
            )
    log(
        f"[Eval] Running with concurrency={args.concurrency}, timeout={args.timeout_sec}s"
    )

    if not sweep_top_k_values:
        summary, elapsed = run_single_eval(
            rows=rows,
            args=args,
            runtime_cfg=runtime_cfg,
            retrieval_top_k=args.retrieval_top_k,
            results_path=Path(args.results),
            summary_path=Path(args.summary),
            confusion_path=Path(args.confusion),
            failures_path=Path(args.failures),
        )
        log(f"[Eval] Completed in {elapsed:.1f}s.")
        log(f"[Eval] Results: {args.results}")
        log(f"[Eval] Summary: {args.summary}")
        log(f"[Eval] Confusion: {args.confusion}")
        log(f"[Eval] Failures: {args.failures}")
        return

    base_results = Path(args.results)
    base_summary = Path(args.summary)
    base_confusion = Path(args.confusion)
    base_failures = Path(args.failures)

    sweep_rows: list[dict[str, Any]] = []
    for top_k in sweep_top_k_values:
        token = f"k{top_k}"
        run_results_path = with_token_suffix(base_results, token)
        run_summary_path = with_token_suffix(base_summary, token)
        run_confusion_path = with_token_suffix(base_confusion, token)
        run_failures_path = with_token_suffix(base_failures, token)

        log(f"[Eval][Sweep] Running top_k={top_k}")
        summary, elapsed = run_single_eval(
            rows=rows,
            args=args,
            runtime_cfg=runtime_cfg,
            retrieval_top_k=top_k,
            results_path=run_results_path,
            summary_path=run_summary_path,
            confusion_path=run_confusion_path,
            failures_path=run_failures_path,
        )
        sweep_rows.append(
            {
                "top_k": top_k,
                "duration_sec": elapsed,
                "results_path": str(run_results_path),
                "summary_path": str(run_summary_path),
                "confusion_path": str(run_confusion_path),
                "failures_path": str(run_failures_path),
                "coverage": summary["sample_counts"]["coverage"],
                "claims_without_evidence": summary["sample_counts"][
                    "claims_without_evidence"
                ],
                "precision": summary["detection"]["precision"],
                "recall": summary["detection"]["recall"],
                "f1": summary["detection"]["f1"],
                "accuracy": summary["detection"]["accuracy"],
                "category_accuracy": summary["category"]["accuracy"],
                "span_exact": summary["localization"]["exact_match"],
                "span_overlap_f1": summary["localization"]["token_overlap_f1"],
            }
        )

    sweep_report = {
        "input": str(input_path),
        "samples": len(rows),
        "model": args.model,
        "evidence_source": runtime_cfg.evidence_source,
        "qdrant_url": args.qdrant_url,
        "qdrant_collection": args.qdrant_collection,
        "tenant_id": args.tenant_id,
        "fallback_to_context": runtime_cfg.fallback_to_context,
        "runs": sweep_rows,
    }
    sweep_report_path = Path(args.sweep_report)
    sweep_report_path.parent.mkdir(parents=True, exist_ok=True)
    with sweep_report_path.open("w", encoding="utf-8") as f:
        json.dump(sweep_report, f, ensure_ascii=False, indent=2)

    log("[Eval][Sweep] Comparison table")
    log(
        "[Eval][Sweep] top_k | f1 | precision | recall | accuracy | claims_without_evidence | duration_s"
    )
    for row in sweep_rows:
        log(
            "[Eval][Sweep] "
            f"{row['top_k']} | {row['f1']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['accuracy']:.4f} | "
            f"{row['claims_without_evidence']} | {row['duration_sec']:.1f}"
        )

    log("[Eval][Sweep] Completed.")
    log(f"[Eval][Sweep] Report: {args.sweep_report}")


if __name__ == "__main__":
    main()
