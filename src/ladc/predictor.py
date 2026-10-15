"""Initial interpretable quality-risk predictor for dynamic budgeting."""

from __future__ import annotations

from ladc.features import PromptFeatures, extract_prompt_features
from ladc.interfaces import Tokenizer


class HeuristicQualityPredictor:
    """Estimate retained quality without using gold answers.

    This is a calibration baseline, not a final learned quality model. The
    formula is deliberately exposed so later experiments can replace it with a
    predictor trained on frozen development results.
    """

    def predict_from_features(self, features: PromptFeatures, retain_ratio: float) -> float:
        if not 0.0 < retain_ratio <= 1.0:
            raise ValueError("retain_ratio must be in the interval (0, 1]")
        compression = 1.0 - retain_ratio
        information_density = 0.65 * features.unique_token_ratio + 0.35 * (1.0 - features.context_query_overlap)
        prefix_risk = 0.5 * features.query_alignment_position + 0.5 * features.tail_query_overlap
        risk = compression * (0.20 + 0.45 * information_density + 0.35 * prefix_risk)
        return max(0.0, min(1.0, 1.0 - risk))

    def predict(self, prompt: str, *, tokenizer: Tokenizer, retain_ratio: float) -> float:
        features = extract_prompt_features(prompt, tokenizer=tokenizer)
        return self.predict_from_features(features, retain_ratio)
