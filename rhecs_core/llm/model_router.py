import os
from typing import Iterable, List, Optional

DEFAULT_MODEL_CANDIDATES = [
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
]


def _has_nonempty_system_instruction(config) -> bool:
    system_instruction = getattr(config, "system_instruction", None)
    if system_instruction is None:
        return False
    if isinstance(system_instruction, str):
        return bool(system_instruction.strip())
    if isinstance(system_instruction, list):
        return any(str(part).strip() for part in system_instruction)
    return True


def _assert_system_instruction(config) -> None:
    if not _has_nonempty_system_instruction(config):
        raise ValueError(
            "GenerateContentConfig.system_instruction must be provided and non-empty before API call."
        )


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _parse_candidates(raw_value: Optional[str]) -> List[str]:
    if not raw_value:
        return []
    return [model.strip() for model in raw_value.split(",") if model.strip()]


def get_model_candidates(
    explicit_candidates: Optional[Iterable[str]] = None,
    env_var: Optional[str] = None,
) -> List[str]:
    if explicit_candidates:
        return _dedupe_keep_order(list(explicit_candidates))

    env_candidates: List[str] = []
    if env_var:
        env_candidates.extend(_parse_candidates(os.getenv(env_var)))

    env_candidates.extend(_parse_candidates(os.getenv("GEMINI_MODEL_CANDIDATES")))
    env_candidates.extend(_parse_candidates(os.getenv("GOOGLE_MODEL_CANDIDATES")))

    if env_candidates:
        return _dedupe_keep_order(env_candidates)

    return list(DEFAULT_MODEL_CANDIDATES)


def _is_retryable_model_error(exc: Exception) -> bool:
    msg = str(exc).upper()
    retry_tokens = [
        "503",
        "UNAVAILABLE",
        "429",
        "404",
        "NOT_FOUND",
        "IS NOT FOUND",
        "RESOURCE_EXHAUSTED",
        "RATE LIMIT",
    ]
    return any(token in msg for token in retry_tokens)


def generate_content_with_fallback(
    client,
    contents,
    config,
    model_candidates: Optional[Iterable[str]] = None,
    env_var: Optional[str] = None,
):
    _assert_system_instruction(config)
    candidates = get_model_candidates(model_candidates, env_var=env_var)
    last_exc: Optional[Exception] = None

    for model_name in candidates:
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            last_exc = exc
            if _is_retryable_model_error(exc):
                continue
            raise

    attempted = ", ".join(candidates)
    raise RuntimeError(
        f"All candidate models failed: [{attempted}]. Last error: {last_exc}"
    ) from last_exc


async def generate_content_with_fallback_async(
    client,
    contents,
    config,
    model_candidates: Optional[Iterable[str]] = None,
    env_var: Optional[str] = None,
):
    _assert_system_instruction(config)
    candidates = get_model_candidates(model_candidates, env_var=env_var)
    last_exc: Optional[Exception] = None

    for model_name in candidates:
        try:
            return await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            last_exc = exc
            if _is_retryable_model_error(exc):
                continue
            raise

    attempted = ", ".join(candidates)
    raise RuntimeError(
        f"All candidate models failed: [{attempted}]. Last error: {last_exc}"
    ) from last_exc
