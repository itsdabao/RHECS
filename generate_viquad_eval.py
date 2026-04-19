import argparse
import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


DEFAULT_MODEL_CANDIDATES = [
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

HALLUCINATION_CATEGORIES = [
    "ENTITY_ERROR",
    "RELATION_ERROR",
    "CONTRADICTORY",
    "UNVERIFIABLE",
    "FABRICATED",
]


@dataclass
class Candidate:
    sample_id: str
    context: str
    question: str
    answer: str
    is_hallucinated: bool
    target_category: Optional[str] = None


class PoisonResult(BaseModel):
    draft_text: str
    fault_span: Optional[str] = None
    error_category: Optional[Literal[
        "ENTITY_ERROR",
        "RELATION_ERROR",
        "CONTRADICTORY",
        "UNVERIFIABLE",
        "FABRICATED",
    ]] = None


@dataclass
class ModelRuntimeState:
    lock: asyncio.Lock
    disabled_models: set[str]
    cooldown_until: dict[str, float]


class ModelFallbackError(RuntimeError):
    def __init__(self, model_errors: dict[str, Exception]):
        self.model_errors = model_errors
        last_model = list(model_errors.keys())[-1] if model_errors else "unknown"
        last_error = model_errors[last_model] if model_errors else RuntimeError("Unknown")
        super().__init__(
            f"Tất cả model fallback đều thất bại. Model cuối: {last_model}. Lỗi: {_short_error(last_error)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RHECS evaluation data from UIT-ViQuAD 2.0 dev.json"
    )
    parser.add_argument(
        "--input",
        default="dev.json",
        help="Path to UIT-ViQuAD 2.0 dev.json (SQuAD 2.0 format)",
    )
    parser.add_argument(
        "--output",
        default="data/eval_rhecs.jsonl",
        help="Output file path (.jsonl by default)",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-lite-preview",
        help="Gemini model chính (được ưu tiên gọi trước).",
    )
    parser.add_argument(
        "--model-candidates",
        default="",
        help="Danh sách model fallback, phân tách bằng dấu phẩy. Ví dụ: gemini-2.5-flash,gemini-2.0-flash,gemini-flash-latest",
    )
    parser.add_argument(
        "--max-context-words",
        type=int,
        default=200,
        help="Keep only contexts with at most this many words",
    )
    parser.add_argument("--n-clean", type=int, default=50, help="Number of clean samples")
    parser.add_argument("--n-error", type=int, default=100, help="Number of hallucinated samples")
    parser.add_argument(
        "--error-categories",
        default=",".join(HALLUCINATION_CATEGORIES),
        help="Danh sách category lỗi cần sinh, cách nhau bằng dấu phẩy.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent Gemini requests",
    )
    parser.add_argument(
        "--output-format",
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format",
    )
    return parser.parse_args()


def load_squad(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_first_non_empty_answer(answer_list: list[dict]) -> Optional[str]:
    for answer_obj in answer_list:
        text = (answer_obj or {}).get("text", "")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


def extract_candidates(dataset: dict, max_context_words: int) -> tuple[list[Candidate], list[Candidate]]:
    clean_candidates: list[Candidate] = []
    error_candidates: list[Candidate] = []

    for article in dataset.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            if not isinstance(context, str) or not context.strip():
                continue

            if len(context.split()) > max_context_words:
                continue

            for qa in paragraph.get("qas", []):
                qa_id = str(qa.get("id", ""))
                question = str(qa.get("question", "")).strip()
                is_impossible = bool(qa.get("is_impossible", False))

                if not qa_id or not question:
                    continue

                if is_impossible:
                    plausible_answers = qa.get("plausible_answers", [])
                    for idx, plausible in enumerate(plausible_answers):
                        answer_text = str((plausible or {}).get("text", "")).strip()
                        if not answer_text:
                            continue
                        error_candidates.append(
                            Candidate(
                                sample_id=f"{qa_id}__plausible_{idx}",
                                context=context,
                                question=question,
                                answer=answer_text,
                                is_hallucinated=True,
                            )
                        )
                else:
                    answer_text = pick_first_non_empty_answer(qa.get("answers", []))
                    if answer_text:
                        clean_candidates.append(
                            Candidate(
                                sample_id=qa_id,
                                context=context,
                                question=question,
                                answer=answer_text,
                                is_hallucinated=False,
                            )
                        )

    return clean_candidates, error_candidates


def sample_candidates(
    clean_pool: list[Candidate],
    error_pool: list[Candidate],
    n_clean: int,
    n_error: int,
    seed: int,
    error_categories: list[str],
) -> list[Candidate]:
    rng = random.Random(seed)

    take_clean = min(n_clean, len(clean_pool))
    take_error = min(n_error, len(error_pool))

    if take_clean < n_clean:
        print(f"[CẢNH BÁO] Yêu cầu {n_clean} mẫu chuẩn, nhưng chỉ có {take_clean} mẫu khả dụng.")
    if take_error < n_error:
        print(f"[CẢNH BÁO] Yêu cầu {n_error} mẫu lỗi, nhưng chỉ có {take_error} mẫu khả dụng.")

    sampled_clean = rng.sample(clean_pool, take_clean)

    sampled_error = rng.sample(error_pool, take_error)
    category_counts = {category: take_error // len(error_categories) for category in error_categories}
    for i in range(take_error % len(error_categories)):
        category_counts[error_categories[i]] += 1

    category_plan: list[str] = []
    for category in error_categories:
        category_plan.extend([category] * category_counts[category])
    rng.shuffle(category_plan)

    assigned_error_samples: list[Candidate] = []
    for base_sample, category in zip(sampled_error, category_plan):
        assigned_error_samples.append(
            Candidate(
                sample_id=base_sample.sample_id,
                context=base_sample.context,
                question=base_sample.question,
                answer=base_sample.answer,
                is_hallucinated=True,
                target_category=category,
            )
        )

    quota_text = ", ".join(f"{k}: {v}" for k, v in category_counts.items())
    print(f"[THÔNG TIN] Quota mẫu lỗi theo category: {quota_text}")

    all_samples = sampled_clean + assigned_error_samples
    rng.shuffle(all_samples)
    return all_samples


def build_prompt(candidate: Candidate) -> str:
        if candidate.is_hallucinated:
                target_category = candidate.target_category or "ENTITY_ERROR"
                return f"""
Bạn là một chuyên gia tạo dữ liệu đánh giá hallucination cho hệ thống NLP.

Nhiệm vụ:
- Hợp nhất câu hỏi và câu trả lời giả (plausible answer) thành 1 câu khẳng định tiếng Việt tự nhiên.
- Câu tạo ra phải SAI so với context.
- Bắt buộc sinh đúng loại lỗi mục tiêu: {target_category}
- Trả về JSON hợp lệ với 3 trường:
    - draft_text: string
    - fault_span: string (cụm sai trong draft_text)
    - error_category: một trong ENTITY_ERROR, RELATION_ERROR, CONTRADICTORY, UNVERIFIABLE, FABRICATED

Định nghĩa category:
- ENTITY_ERROR: Sai thực thể (người, địa danh, tổ chức, đối tượng).
- RELATION_ERROR: Đúng thực thể nhưng sai quan hệ/hành động/tính chất.
- CONTRADICTORY: Nội dung mâu thuẫn trực tiếp với context.
- UNVERIFIABLE: Khẳng định không đủ bằng chứng từ context để xác minh.
- FABRICATED: Bịa thêm thông tin không có trong context.

Ràng buộc:
- Không được trả markdown.
- Không được bổ sung key ngoài schema.
- fault_span phải xuất hiện nguyên văn trong draft_text.
- error_category phải đúng chính xác là: {target_category}

[CONTEXT]
{candidate.context}

[QUESTION]
{candidate.question}

[PLAUSIBLE_ANSWER]
{candidate.answer}
""".strip()

        return f"""
Bạn là một chuyên gia tạo dữ liệu đánh giá factual consistency.

Nhiệm vụ:
- Hợp nhất câu hỏi và câu trả lời đúng thành 1 câu khẳng định tiếng Việt tự nhiên.
- Câu tạo ra phải đúng theo context.
- Trả về JSON hợp lệ với 3 trường:
    - draft_text: string
    - fault_span: null
    - error_category: null

Ràng buộc:
- Không được trả markdown.
- Không được bổ sung key ngoài schema.

[CONTEXT]
{candidate.context}

[QUESTION]
{candidate.question}

[ANSWER]
{candidate.answer}
""".strip()


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen = set()
    output: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _parse_csv_models(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def parse_error_categories(raw_value: str) -> list[str]:
    categories = _parse_csv_models(raw_value)
    if not categories:
        return list(HALLUCINATION_CATEGORIES)

    normalized = []
    for category in categories:
        c = category.strip().upper()
        if c not in HALLUCINATION_CATEGORIES:
            raise ValueError(
                f"Category không hợp lệ: {category}. Hợp lệ: {', '.join(HALLUCINATION_CATEGORIES)}"
            )
        normalized.append(c)

    return _dedupe_keep_order(normalized)


def get_model_candidates(primary_model: str, model_candidates_arg: str) -> list[str]:
    models: list[str] = [primary_model]
    models.extend(_parse_csv_models(model_candidates_arg))

    env_models = os.getenv("GEMINI_MODEL_CANDIDATES") or os.getenv("GOOGLE_MODEL_CANDIDATES")
    models.extend(_parse_csv_models(env_models or ""))

    models.extend(DEFAULT_MODEL_CANDIDATES)
    return _dedupe_keep_order(models)


def _error_message(exc: Exception) -> str:
    return str(exc)


def _short_error(exc: Exception) -> str:
    message = _error_message(exc).splitlines()[0].strip()
    return message[:300]


def _is_fallback_error(exc: Exception) -> bool:
    message = _error_message(exc).upper()
    tokens = [
        "429",
        "503",
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "RATE LIMIT",
        "NOT_FOUND",
        "404",
    ]
    return any(token in message for token in tokens)


def _is_daily_quota_error(exc: Exception) -> bool:
    message = _error_message(exc).upper().replace(" ", "")
    return "GENERATEREQUESTSPERDAYPERPROJECTPERMODEL" in message


def _extract_retry_delay_seconds(exc: Exception) -> int:
    message = _error_message(exc)

    match = re.search(r"retryDelay'\s*:\s*'(\d+)s'", message)
    if match:
        return int(match.group(1))

    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", message, flags=re.IGNORECASE)
    if match:
        return int(float(match.group(1)))

    return 30


def _should_retry_exception(exc: Exception) -> bool:
    if isinstance(exc, ModelFallbackError):
        if not exc.model_errors:
            return True
        return any(not _is_daily_quota_error(err) for err in exc.model_errors.values())

    if _is_daily_quota_error(exc):
        return False
    return True


@retry(
    retry=retry_if_exception(_should_retry_exception),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def call_gemini(client: genai.Client, model_candidates: list[str], candidate: Candidate) -> PoisonResult:
    prompt = build_prompt(candidate)

    model_errors: dict[str, Exception] = {}
    response = None

    for model_name in model_candidates:
        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=PoisonResult,
                ),
            )
            break
        except Exception as exc:
            model_errors[model_name] = exc
            if _is_fallback_error(exc):
                continue
            raise

    if response is None:
        raise ModelFallbackError(model_errors)

    try:
        result = PoisonResult.model_validate_json(response.text)
    except ValidationError as exc:
        raise ValueError(f"JSON trả về từ model không đúng schema: {response.text}") from exc

    draft_text = (result.draft_text or "").strip()
    if not draft_text:
        raise ValueError("Model trả về draft_text rỗng")

    if candidate.is_hallucinated:
        if not result.fault_span or not result.fault_span.strip():
            raise ValueError("Mẫu hallucinated thiếu fault_span")
        if result.error_category not in HALLUCINATION_CATEGORIES:
            raise ValueError("Mẫu hallucinated thiếu error_category hợp lệ")
        if candidate.target_category and result.error_category != candidate.target_category:
            raise ValueError(
                f"Model trả về sai category. Yêu cầu: {candidate.target_category}, nhận: {result.error_category}"
            )
        if result.fault_span not in result.draft_text:
            raise ValueError("Không tìm thấy fault_span trong draft_text")
    else:
        result.fault_span = None
        result.error_category = None

    return result


async def process_candidate(
    client: genai.Client,
    model_candidates: list[str],
    runtime_state: ModelRuntimeState,
    semaphore: asyncio.Semaphore,
    candidate: Candidate,
) -> dict:
    async with semaphore:
        async def pick_available_models() -> list[str]:
            now = time.time()
            async with runtime_state.lock:
                available = []
                for name in model_candidates:
                    if name in runtime_state.disabled_models:
                        continue
                    cooldown_deadline = runtime_state.cooldown_until.get(name, 0)
                    if cooldown_deadline > now:
                        continue
                    available.append(name)
                return available

        active_models = await pick_available_models()
        if not active_models:
            raise RuntimeError("Không còn model khả dụng (đều đang bị quota/cooldown).")

        try:
            result = await call_gemini(client, active_models, candidate)
        except ModelFallbackError as exc:
            async with runtime_state.lock:
                for name, model_exc in exc.model_errors.items():
                    if _is_daily_quota_error(model_exc):
                        runtime_state.disabled_models.add(name)
                    elif _is_fallback_error(model_exc):
                        delay_seconds = _extract_retry_delay_seconds(model_exc)
                        runtime_state.cooldown_until[name] = time.time() + max(delay_seconds, 3)
            raise
        except Exception as exc:
            if _is_fallback_error(exc):
                delay_seconds = _extract_retry_delay_seconds(exc)
                async with runtime_state.lock:
                    for name in active_models:
                        if _is_daily_quota_error(exc):
                            runtime_state.disabled_models.add(name)
                        else:
                            runtime_state.cooldown_until[name] = time.time() + max(delay_seconds, 3)
            raise

        return {
            "id": candidate.sample_id,
            "context": candidate.context,
            "draft": result.draft_text,
            "is_hallucinated": candidate.is_hallucinated,
            "expected_fault_span": result.fault_span,
            "category": result.error_category,
        }


def write_output(records: list[dict], output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu GEMINI_API_KEY hoặc GOOGLE_API_KEY trong môi trường.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {input_path}")

    dataset = load_squad(input_path)
    clean_pool, error_pool = extract_candidates(dataset, args.max_context_words)
    error_categories = parse_error_categories(args.error_categories)

    print(f"[THÔNG TIN] Số mẫu ứng viên chuẩn: {len(clean_pool)}")
    print(f"[THÔNG TIN] Số mẫu ứng viên lỗi: {len(error_pool)}")

    selected = sample_candidates(
        clean_pool=clean_pool,
        error_pool=error_pool,
        n_clean=args.n_clean,
        n_error=args.n_error,
        seed=args.seed,
        error_categories=error_categories,
    )

    print(f"[THÔNG TIN] Số mẫu được chọn: {len(selected)}")

    client = genai.Client(api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    model_candidates = get_model_candidates(args.model, args.model_candidates)
    print(f"[THÔNG TIN] Danh sách model fallback: {', '.join(model_candidates)}")
    runtime_state = ModelRuntimeState(
        lock=asyncio.Lock(),
        disabled_models=set(),
        cooldown_until={},
    )

    tasks = [
        asyncio.create_task(process_candidate(client, model_candidates, runtime_state, semaphore, item))
        for item in selected
    ]

    records: list[dict] = []
    total = len(tasks)
    done = 0

    for finished in asyncio.as_completed(tasks):
        try:
            records.append(await finished)
        except Exception as exc:
            print(f"[LỖI] Mẫu thất bại sau nhiều lần retry: {_short_error(exc)}")
        done += 1
        if done % 10 == 0 or done == total:
            print(f"[TIẾN ĐỘ] Đã xử lý {done}/{total} mẫu...")

    write_output(records, Path(args.output), args.output_format)
    category_counts: dict[str, int] = {}
    for record in records:
        category = record.get("category")
        key = str(category) if category is not None else "CLEAN"
        category_counts[key] = category_counts.get(key, 0) + 1
    print(f"[THỐNG KÊ] Phân bố category đã ghi: {category_counts}")
    print(f"[HOÀN TẤT] Đã ghi {len(records)} bản ghi vào {args.output}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
