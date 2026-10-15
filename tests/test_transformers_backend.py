import pytest

from ladc.inference.transformers_backend import TransformersInferenceBackend


def test_transformers_backend_validates_generation_budget():
    with pytest.raises(ValueError, match="max_new_tokens"):
        TransformersInferenceBackend(
            model=object(),
            tokenizer=object(),
            device="cpu",
            max_new_tokens=0,
        )
