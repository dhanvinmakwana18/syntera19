import pytest
import os
from unittest.mock import patch, MagicMock
from intelligence.contracts import IntelligenceRequest, StructuredGenerationRequest
from providers.nvidia_provider import NVIDIAProvider

# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def mock_env():
    # Safely mock environment so we never use the real key in tests
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "fake_test_key"}):
        yield

@pytest.fixture
def mock_openai():
    with patch("providers.nvidia_provider.OpenAI") as mock:
        yield mock

# ── Tests ─────────────────────────────────────────────────

def test_nvidia_provider_initialization(mock_env, mock_openai):
    provider = NVIDIAProvider(model_name="test-model")
    assert provider.profile.name == "test-model"
    assert provider.profile.provider == "nvidia"
    assert provider.profile.cost_tier == "frontier"
    
    mock_openai.assert_called_once_with(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="fake_test_key"
    )

def test_nvidia_provider_missing_key():
    with patch.dict(os.environ, clear=True):
        with pytest.raises(ValueError, match="NVIDIA_API_KEY not found"):
            NVIDIAProvider()

def test_nvidia_provider_generate(mock_env, mock_openai):
    provider = NVIDIAProvider(model_name="test-model")
    
    # Mock the API response
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "This is a test response."
    provider._client.chat.completions.create.return_value = mock_completion
    
    request = IntelligenceRequest(prompt="Hello", temperature=0.5, max_tokens=100)
    response = provider.generate(request)
    
    assert response.success is True
    assert response.content == "This is a test response."
    assert response.model == "test-model"
    assert response.provider == "nvidia"
    
    # Verify the configuration matches expected kwargs
    provider._client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.5,
        max_tokens=100,
        top_p=0.95,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}}
    )

def test_nvidia_provider_structured_generate(mock_env, mock_openai):
    from pydantic import BaseModel
    
    class TestSchema(BaseModel):
        answer: str
        confidence: float
        
    provider = NVIDIAProvider(model_name="test-model")
    
    # Mock a valid JSON response
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = '{"answer": "42", "confidence": 0.99}'
    provider._client.chat.completions.create.return_value = mock_completion
    
    request = StructuredGenerationRequest(prompt="What is life?", output_schema=TestSchema)
    response = provider.structured_generate(request)
    
    assert response.success is True
    assert response.data.answer == "42"
    assert response.data.confidence == 0.99
    
    # Verify JSON mode and schema handling
    provider._client.chat.completions.create.assert_called_once()
    kwargs = provider._client.chat.completions.create.call_args[1]
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "JSON" in kwargs["messages"][-1]["content"]

def test_nvidia_provider_error_handling(mock_env, mock_openai):
    provider = NVIDIAProvider(model_name="test-model")
    
    # Simulate a network error
    provider._client.chat.completions.create.side_effect = Exception("Network timeout")
    
    request = IntelligenceRequest(prompt="Fail me")
    response = provider.generate(request)
    
    assert response.success is False
    assert "Network timeout" in response.error
