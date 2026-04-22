from rhecs_core.runtime import RuntimeConfig, VerificationStrategy


def _clear_runtime_env(monkeypatch) -> None:
    keys = [
        "RHECS_VERIFICATION_STRATEGY",
        "RHECS_DEFAULT_STRATEGY",
        "RHECS_ROUTER_MAX_RECURSION",
        "RHECS_MAX_RECURSION",
        "RHECS_ROUTER_MAX_CALLS",
        "RHECS_CALL_BUDGET",
        "RHECS_ROUTER_MAX_TOKENS",
        "RHECS_TOKEN_BUDGET",
        "RHECS_ROUTER_TIMEOUT_MS",
        "RHECS_TIMEOUT_BUDGET_MS",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_runtime_config_from_env_defaults(monkeypatch):
    _clear_runtime_env(monkeypatch)

    cfg = RuntimeConfig.from_env()

    assert cfg.default_strategy == VerificationStrategy.DIRECT_LLM
    assert cfg.max_recursion == 2
    assert cfg.call_budget == 10
    assert cfg.token_budget == 5000
    assert cfg.timeout_budget == 30000


def test_runtime_config_from_env_overrides(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("RHECS_VERIFICATION_STRATEGY", "rlm_recursive")
    monkeypatch.setenv("RHECS_ROUTER_MAX_RECURSION", "3")
    monkeypatch.setenv("RHECS_ROUTER_MAX_CALLS", "12")
    monkeypatch.setenv("RHECS_ROUTER_MAX_TOKENS", "6000")
    monkeypatch.setenv("RHECS_ROUTER_TIMEOUT_MS", "45000")

    cfg = RuntimeConfig.from_env()

    assert cfg.default_strategy == VerificationStrategy.RLM_RECURSIVE
    assert cfg.max_recursion == 3
    assert cfg.call_budget == 12
    assert cfg.token_budget == 6000
    assert cfg.timeout_budget == 45000


def test_runtime_config_from_env_fallback_keys(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("RHECS_DEFAULT_STRATEGY", "rlm_recursive")
    monkeypatch.setenv("RHECS_MAX_RECURSION", "4")
    monkeypatch.setenv("RHECS_CALL_BUDGET", "20")
    monkeypatch.setenv("RHECS_TOKEN_BUDGET", "9000")
    monkeypatch.setenv("RHECS_TIMEOUT_BUDGET_MS", "55000")

    cfg = RuntimeConfig.from_env()

    assert cfg.default_strategy == VerificationStrategy.RLM_RECURSIVE
    assert cfg.max_recursion == 4
    assert cfg.call_budget == 20
    assert cfg.token_budget == 9000
    assert cfg.timeout_budget == 55000


def test_runtime_config_invalid_budget_raises(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("RHECS_ROUTER_MAX_CALLS", "0")

    try:
        RuntimeConfig.from_env()
    except ValueError as exc:
        assert "RHECS_ROUTER_MAX_CALLS" in str(exc)
        return

    raise AssertionError("Expected ValueError for invalid budget env value")
