import os
import httpx
from pathlib import Path

FABRIC_BASE_URL = "https://raw.githubusercontent.com/danielmiessler/fabric/main/patterns"
CACHE_DIR = Path.home() / ".syntera" / "patterns"

class FabricPatternManager:
    @staticmethod
    def get_pattern(pattern_name: str) -> str:
        """Fetch a system prompt pattern from Daniel Miessler's Fabric repo."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{pattern_name}.md"
        
        # Check local cache first
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
            
        # If not cached, fetch from GitHub
        url = f"{FABRIC_BASE_URL}/{pattern_name}/system.md"
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                content = response.text
                cache_file.write_text(content, encoding="utf-8")
                return content
            else:
                return FabricPatternManager._get_fallback_pattern(pattern_name)
        except Exception:
            return FabricPatternManager._get_fallback_pattern(pattern_name)
            
    @staticmethod
    def _get_fallback_pattern(name: str) -> str:
        if "test" in name.lower():
            return (
                "You are an expert Python QA architect. Your job is to write clean, modular, "
                "comprehensive pytest unit tests. Output only valid Python code inside markdown blocks."
            )
        return (
            "You are an expert Python architect. Your job is to write clean, modular, "
            "production-ready code. Always include type hints, comprehensive docstrings, "
            "and ensure it adheres to PEP 8 standards. Output only valid Python code inside "
            "markdown blocks."
        )
