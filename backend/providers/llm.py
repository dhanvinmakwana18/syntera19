import requests
import os
import time
from core.config import settings

class LLMProvider:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.ollama_base_url = "http://localhost:11434/api"
        self.model = settings.LLM_MODEL
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # External generic config
        self.astra_api_key = settings.ASTRA_API_KEY or os.getenv("ASTRA_API_KEY")
        self.openai_api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        
        # If a base URL is specified, use it. Otherwise for 'openai', fallback to standard.
        self.base_url = settings.LLM_BASE_URL
        if self.provider == "openai" and not self.base_url:
            self.base_url = "https://api.openai.com/v1/chat/completions"
            
        self.max_retries = settings.ASTRA_MAX_RETRIES

    def _safe_request(self, url, headers, payload, timeout=60):
        """Executes a request with exponential backoff and sanitizes errors."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                return response.json(), (time.time() - start_time) * 1000
            except requests.exceptions.RequestException as e:
                # Sanitize authorization headers in error messages
                if hasattr(e, 'request') and e.request:
                    if 'Authorization' in e.request.headers:
                        e.request.headers['Authorization'] = 'REDACTED'
                last_err = str(e)
                # Ensure the actual secret string doesn't leak in the text of the error
                if self.astra_api_key:
                    last_err = last_err.replace(self.astra_api_key, 'REDACTED')
                if self.openai_api_key:
                    last_err = last_err.replace(self.openai_api_key, 'REDACTED')
                    
                time.sleep(2 ** attempt)  # Exponential backoff
        raise Exception(f"Max retries ({self.max_retries}) exceeded. Last error: {last_err}")

    def generate(self, prompt: str, system_prompt: str = None, json_mode: bool = False, provider_override: str = None) -> str:
        active_provider = (provider_override or self.provider).lower()
        
        if active_provider in ["astra", "openai"]:
            key = self.astra_api_key if active_provider == "astra" else self.openai_api_key
            if not key:
                raise ValueError(f"API key is required for {active_provider} provider.")
            if not self.base_url:
                raise ValueError(f"LLM_BASE_URL must be defined for {active_provider} integration.")
                
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": settings.ASTRA_MODEL if active_provider == "astra" else self.model,
                "messages": [],
                "temperature": settings.ASTRA_TEMPERATURE
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
                
            if system_prompt:
                payload["messages"].append({"role": "system", "content": system_prompt})
            payload["messages"].append({"role": "user", "content": prompt})
            
            try:
                data, latency = self._safe_request(self.base_url, headers, payload)
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                else:
                    return str(data)
            except Exception as e:
                print(f"External API Error: {e}")
                return f"Error response: {e}"

        elif active_provider == "gemini" or self.gemini_api_key:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
            payload = {
                "contents": [{"parts": [{"text": (system_prompt + "\n\n" if system_prompt else "") + prompt}]}]
            }
            try:
                data, _ = self._safe_request(url, {"Content-Type": "application/json"}, payload)
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                print(f"Gemini API Error: {e}")
                return f"Error response: {e}"
                
        # Default fallback to Local Ollama Provider
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        if json_mode:
            payload["format"] = "json"
            
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            data, _ = self._safe_request(f"{self.ollama_base_url}/generate", {"Content-Type": "application/json"}, payload)
            return data.get("response", "")
        except Exception as e:
            if "Connection refused" in str(e):
                print("LLM Error: Connection refused. Is Ollama running on localhost:11434?")
                print("Fallback: Using REAL local transformers model since Ollama is unavailable.")
                return self._generate_local_hf(prompt, system_prompt)
            print(f"LLM Error: {e}")
            return f"Error response: {e}"

    def _generate_local_hf(self, prompt: str, system_prompt: str = None) -> str:
        if not hasattr(self, "hf_pipeline"):
            from transformers import pipeline
            import torch
            print("Loading fallback HuggingFace LLM (Qwen/Qwen2.5-0.5B-Instruct)...")
            self.hf_pipeline = pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-0.5B-Instruct",
                device="cpu",
                torch_dtype=torch.float32
            )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            out = self.hf_pipeline(messages, max_new_tokens=256, do_sample=False)
            return out[0]["generated_text"][-1]["content"]
        except Exception as e:
            print(f"Local HF LLM Error: {e}")
            return f"Error: {e}"

# -------------------------------------------------------------
# PHASE 4: Component Registration
# -------------------------------------------------------------
from core.registry import registry
from typing import Any

def create_llm_provider() -> Any:
    return LLMProvider()

registry.register_llm("default", create_llm_provider)
