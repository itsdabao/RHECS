from rhecs_core.query_strategy import (
    QueryRequest,
    QueryRouter,
    QueryStrategy,
    RLMBridge,
)


def test_query_strategy_package_import_smoke():
    request = QueryRequest(
        original_sentence="A is B",
        claim={"subject": "A", "predicate": "is", "object": "B"},
        strategy=QueryStrategy.DIRECT_LLM,
    )

    assert request.strategy == QueryStrategy.DIRECT_LLM
    assert isinstance(QueryRouter(), QueryRouter)
    assert isinstance(RLMBridge(), RLMBridge)
