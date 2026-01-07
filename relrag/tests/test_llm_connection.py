from relrag.utils.llm_client import LLMChatClient, SERVED_MODEL_NAME, VLLM_ENDPOINT


def test_llm_connection_default_override(monkeypatch):
    monkeypatch.delenv("RELRAG_ALLOW_CUSTOM_LLM", raising=False)
    client = LLMChatClient(endpoint="http://custom:1234", model="custom-model")
    assert client.endpoint == VLLM_ENDPOINT
    assert client.model == SERVED_MODEL_NAME


def test_llm_connection_allow_custom(monkeypatch):
    monkeypatch.setenv("RELRAG_ALLOW_CUSTOM_LLM", "1")
    custom_endpoint = "http://custom:1234"
    custom_model = "custom-model"
    client = LLMChatClient(endpoint=custom_endpoint, model=custom_model)
    assert client.endpoint == custom_endpoint
    assert client.model == custom_model
