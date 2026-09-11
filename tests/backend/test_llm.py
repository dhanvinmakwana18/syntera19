import pytest
import os
from unittest.mock import patch, MagicMock
import requests
from providers.llm import LLMProvider

@pytest.fixture
def mock_settings(monkeypatch):
    from core.config import settings
    settings.LLM_PROVIDER = "astra"
    settings.LLM_BASE_URL = "https://api.mock-astra.com/v1/chat"
    settings.ASTRA_API_KEY = "test_key_123"
    settings.ASTRA_MODEL = "gpt-6-astra"
    settings.ASTRA_MAX_RETRIES = 2

def test_llm_initialization(mock_settings):
    from core.config import settings
    # Force reload or manually init
    provider = LLMProvider()
    assert provider.provider == "astra"
    assert provider.astra_api_key == "test_key_123"

@patch('requests.post')
def test_successful_api_call(mock_post, mock_settings):
    provider = LLMProvider()
    provider.astra_api_key = "test_key_123"
    provider.base_url = "https://api.mock-astra.com/v1/chat"
    provider.provider = "astra"
    
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "Mocked structured response"}}
        ]
    }
    mock_post.return_value = mock_response

    response = provider.generate("What is Syntera?", system_prompt="You are AI.")
    assert response == "Mocked structured response"
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert "Authorization" in kwargs["headers"]
    assert kwargs["headers"]["Authorization"] == "Bearer test_key_123"

@patch('requests.post')
def test_retry_on_failure(mock_post, mock_settings):
    provider = LLMProvider()
    provider.astra_api_key = "test_key_123"
    provider.base_url = "https://api.mock-astra.com/v1/chat"
    provider.provider = "astra"
    provider.max_retries = 2
    
    # Setup mock to fail first, then succeed
    mock_fail = MagicMock()
    mock_fail.raise_for_status.side_effect = requests.exceptions.RequestException("Timeout")
    
    mock_success = MagicMock()
    mock_success.raise_for_status.return_value = None
    mock_success.json.return_value = {
        "choices": [{"message": {"content": "Success after retry"}}]
    }
    
    mock_post.side_effect = [mock_fail, mock_success]

    # Patch time.sleep to not wait during tests
    with patch('time.sleep', return_value=None):
        response = provider.generate("Test retry")
        assert response == "Success after retry"
        assert mock_post.call_count == 2

@patch('requests.post')
def test_api_key_redaction_on_error(mock_post, mock_settings):
    provider = LLMProvider()
    provider.astra_api_key = "secret_key_that_must_not_be_logged"
    provider.base_url = "https://api.mock-astra.com/v1/chat"
    provider.provider = "astra"
    provider.max_retries = 1
    
    mock_fail = MagicMock()
    mock_fail.raise_for_status.side_effect = requests.exceptions.RequestException("Error with secret_key_that_must_not_be_logged")
    mock_post.side_effect = [mock_fail]

    with patch('time.sleep', return_value=None):
        response = provider.generate("Test error")
        assert "secret_key_that_must_not_be_logged" not in response
        assert "REDACTED" in response
