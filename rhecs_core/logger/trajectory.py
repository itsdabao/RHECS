import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any


class TrajectoryLogger:
    def __init__(self, request_id: str, base_dir: str = "artifacts/trajectory"):
        date_dir = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.file_path = os.path.join(base_dir, date_dir, f"{request_id}.jsonl")
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def _to_serializable(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        if hasattr(obj, "value"):
            return obj.value
        return obj

    def _write_line(self, payload: dict) -> None:
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(self._to_serializable(payload), ensure_ascii=False)
                    + "\n"
                )
        except Exception:
            # Logging must never break pipeline execution.
            pass

    def log_metadata(self, request_id: str, tenant_id: str | None) -> None:
        self._write_line(
            {
                "type": "metadata",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "tenant_id": tenant_id,
            }
        )

    def log_transition(self, transition: Any) -> None:
        self._write_line(
            {
                "type": "transition",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "transition": transition,
            }
        )

    def log_error(self, error_info: Any) -> None:
        self._write_line(
            {
                "type": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": error_info,
            }
        )

    def log_summary(
        self,
        request_state: str,
        claims_extracted: int,
        faults_found: int,
        patches_applied: int,
        runtime_errors: int,
    ) -> None:
        self._write_line(
            {
                "type": "summary",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_state": request_state,
                "claims_extracted": claims_extracted,
                "faults_found": faults_found,
                "patches_applied": patches_applied,
                "runtime_errors": runtime_errors,
            }
        )
