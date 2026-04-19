import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải và chuyển UIT-ViQuAD2.0 sang định dạng dev.json (SQuAD 2.0)."
    )
    parser.add_argument(
        "--dataset",
        default="taidng/UIT-ViQuAD2.0",
        help="Tên dataset trên Hugging Face.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Split cần chuyển đổi: train, validation hoặc test.",
    )
    parser.add_argument(
        "--output",
        default="dev.json",
        help="Đường dẫn file đầu ra JSON.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Giới hạn số dòng để test nhanh. 0 nghĩa là dùng toàn bộ split.",
    )
    return parser.parse_args()


def normalize_answer_field(field: Any) -> list[dict[str, Any]]:
    if field is None:
        return []

    if isinstance(field, dict):
        texts = field.get("text", []) or []
        starts = field.get("answer_start", []) or []
        output = []
        for i, text in enumerate(texts):
            text_value = str(text).strip()
            if not text_value:
                continue
            start_value = starts[i] if i < len(starts) else -1
            output.append({"text": text_value, "answer_start": int(start_value)})
        return output

    if isinstance(field, list):
        output = []
        for item in field:
            if not isinstance(item, dict):
                continue
            text_value = str(item.get("text", "")).strip()
            if not text_value:
                continue
            start_value = item.get("answer_start", -1)
            output.append({"text": text_value, "answer_start": int(start_value)})
        return output

    return []


def convert_split(dataset_name: str, split: str, max_rows: int = 0) -> dict[str, Any]:
    ds = load_dataset(dataset_name, split=split)
    if max_rows > 0:
        ds = ds.select(range(min(max_rows, len(ds))))

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for idx, row in enumerate(ds):
        title = str(row.get("title", "") or "untitled")
        context = str(row.get("context", "") or "").strip()
        question = str(row.get("question", "") or "").strip()
        if not context or not question:
            continue

        qa_id = str(row.get("id", "") or f"auto_{idx}")
        is_impossible = bool(row.get("is_impossible", False))

        answers = normalize_answer_field(row.get("answers"))
        plausible_answers = normalize_answer_field(row.get("plausible_answers"))

        qa_obj = {
            "id": qa_id,
            "question": question,
            "is_impossible": is_impossible,
            "answers": answers,
            "plausible_answers": plausible_answers,
        }

        grouped[title][context].append(qa_obj)

    data = []
    for title, context_map in grouped.items():
        paragraphs = []
        for context, qas in context_map.items():
            paragraphs.append({"context": context, "qas": qas})
        data.append({"title": title, "paragraphs": paragraphs})

    return {
        "version": "UIT-ViQuAD2.0-converted",
        "dataset": dataset_name,
        "split": split,
        "data": data,
    }


def main() -> None:
    args = parse_args()
    print(f"[THÔNG TIN] Đang tải dataset: {args.dataset} | split: {args.split}")

    converted = convert_split(args.dataset, args.split, args.max_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False)

    paragraph_count = sum(len(article["paragraphs"]) for article in converted["data"])
    qa_count = sum(len(paragraph["qas"]) for article in converted["data"] for paragraph in article["paragraphs"])

    print(f"[HOÀN TẤT] Đã ghi file: {output_path}")
    print(f"[THỐNG KÊ] Số article: {len(converted['data'])}")
    print(f"[THỐNG KÊ] Số paragraph: {paragraph_count}")
    print(f"[THỐNG KÊ] Số QA: {qa_count}")


if __name__ == "__main__":
    main()
