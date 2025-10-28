from src.eval.answer_similarity import (
    EmbeddingCosineStrategy,
    LLMJudgementStrategy,
    TokenF1Strategy,
)


class StaticEmbeddingProvider:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed(self, text: str):
        return self.mapping.get(text, [0.0, 0.0])


class StubLLMScore:
    def measure(self, test_case, *args, **kwargs):
        return 1.0 if test_case.expected_output == test_case.actual_output else 0.5


def test_token_f1_strategy_scores_overlap():
    strategy = TokenF1Strategy()
    score = strategy.compute("Alpha Beta", "beta gamma")
    assert 0.0 < score < 1.0
    assert strategy.compute("Same words", "same words") == 1.0


def test_embedding_strategy_cosine_similarity():
    provider = StaticEmbeddingProvider(
        {
            "contract": [1.0, 0.0],
            "agreement": [0.9, 0.1],
            "different": [0.0, 1.0],
        }
    )
    strategy = EmbeddingCosineStrategy(provider=provider)
    close_score = strategy.compute("contract", "agreement")
    far_score = strategy.compute("contract", "different")
    assert close_score > far_score
    assert 0.0 <= far_score <= close_score <= 1.0


def test_llm_judgement_strategy_uses_scorer():
    strategy = LLMJudgementStrategy(scorer=StubLLMScore())
    assert strategy.compute("expected", "expected") == 1.0
    assert strategy.compute("expected", "other") == 0.5
