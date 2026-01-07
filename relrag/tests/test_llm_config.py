import os
import pytest
from relrag.utils.llm_client import _normalize_endpoint, _normalize_model, VLLM_ENDPOINT, SERVED_MODEL_NAME

@pytest.fixture
def clean_env():
    old = os.environ.get("RELRAG_ALLOW_CUSTOM_LLM")
    if "RELRAG_ALLOW_CUSTOM_LLM" in os.environ:
        del os.environ["RELRAG_ALLOW_CUSTOM_LLM"]
    yield
    if old is not None:
        os.environ["RELRAG_ALLOW_CUSTOM_LLM"] = old
    else:
        if "RELRAG_ALLOW_CUSTOM_LLM" in os.environ:
            del os.environ["RELRAG_ALLOW_CUSTOM_LLM"]

def test_default_behavior(clean_env):
    # Default: force override
    assert _normalize_endpoint("http://custom:1234") == VLLM_ENDPOINT
    assert _normalize_model("custom-model") == SERVED_MODEL_NAME
    assert _normalize_endpoint(None) == VLLM_ENDPOINT
    assert _normalize_model(None) == SERVED_MODEL_NAME

def test_custom_behavior(clean_env):
    # Custom: allow pass-through
    os.environ["RELRAG_ALLOW_CUSTOM_LLM"] = "1"
    
    custom_ep = "http://custom:1234"
    custom_model = "custom-model"
    
    assert _normalize_endpoint(custom_ep) == custom_ep
    assert _normalize_model(custom_model) == custom_model
    
    # Still handles None/empty
    assert _normalize_endpoint(None) == VLLM_ENDPOINT
    assert _normalize_model("") == SERVED_MODEL_NAME
