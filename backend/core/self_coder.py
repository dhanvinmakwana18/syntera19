import re
import os
import time
import subprocess
from pathlib import Path
from typing import Tuple

from intelligence.core import IntelligenceCore
from intelligence.contracts import TaskComplexity
from intelligence.patterns import FabricPatternManager

class SelfCoder:
    def __init__(self, core: IntelligenceCore):
        self.core = core
        self.backend_dir = Path(__file__).parent.parent
        self.agents_dir = self.backend_dir / "agents"
        self.tests_dir = self.backend_dir / "tests" / "agents"
        
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.tests_dir.mkdir(parents=True, exist_ok=True)

        # Ensure __init__.py exists in agents
        (self.agents_dir / "__init__.py").touch(exist_ok=True)
        (self.tests_dir / "__init__.py").touch(exist_ok=True)

    def _extract_code(self, markdown: str) -> str:
        """Extract python code from a markdown block, perfectly preserving internal docstrings."""
        lines = markdown.splitlines()
        
        # Find opening ```
        start_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                start_idx = i
                break
                
        if start_idx == -1:
            return markdown.strip()
            
        # Find closing ``` (if it exists and wasn't truncated)
        end_idx = -1
        for i in range(len(lines) - 1, start_idx, -1):
            if lines[i].strip().startswith("```"):
                end_idx = i
                break
                
        if end_idx != -1:
            code_lines = lines[start_idx + 1 : end_idx]
        else:
            code_lines = lines[start_idx + 1 :]
            
        return "\n".join(code_lines).strip()

    def _extract_filename(self, prompt: str) -> str:
        """Use a fast local call to generate a valid python filename from the prompt."""
        res = self.core.generate(
            prompt=f"Convert this idea into a short snake_case python filename (ending in .py). Idea: {prompt}. Return ONLY the filename, nothing else.",
            complexity=TaskComplexity.SIMPLE
        )
        name = res.content.strip().lower().replace(" ", "_").replace('"', '').replace("'", "")
        name = re.sub(r'[^a-z0-9_.]', '', name)
        if not name.endswith(".py"):
            name += ".py"
        return name

    def generate_agent(self, prompt: str, target_name: str = None) -> Tuple[Path, Path, str]:
        """Generates an agent, formats it, generates tests, and returns paths."""
        
        # 1. Fetch Fabric Pattern (write_microservices is a common Fabric coding pattern)
        system_prompt = FabricPatternManager.get_pattern("write_microservices")
        
        # 2. Determine Filename
        filename = target_name if target_name else self._extract_filename(prompt)
        if not filename.endswith(".py"): filename += ".py"
        
        file_path = self.agents_dir / filename
        test_filename = f"test_{filename}"
        test_file_path = self.tests_dir / test_filename
        
        # 3. Generate Main Code
        build_prompt = f"Write a complete, production-ready Python module for the following requirement:\n\n{prompt}\n\nInclude necessary imports. Assume it's in a package called 'backend.agents'. Return the python code in a single markdown block."
        
        res = self.core.generate(
            prompt=build_prompt,
            system_prompt=system_prompt,
            complexity=TaskComplexity.COMPLEX
        )
        if not res.success:
            raise Exception(f"Failed to generate code: {res.error}")
            
        code = self._extract_code(res.content)
        file_path.write_text(code, encoding="utf-8")
        
        # Rate limit protection: sleep before generating tests
        time.sleep(5)
        
        # 4. Generate Tests
        test_system_prompt = FabricPatternManager.get_pattern("write_tests")
        test_prompt = f"Write comprehensive pytest unit tests for the following code. The code is located at `backend.agents.{filename[:-3]}`. You must import the necessary classes from that module.\n\nCode:\n```python\n{code}\n```\n\nReturn the test code in a single markdown block."
        
        test_res = self.core.generate(
            prompt=test_prompt,
            system_prompt=test_system_prompt,
            complexity=TaskComplexity.COMPLEX
        )
        if test_res.success:
            test_code = self._extract_code(test_res.content)
            test_file_path.write_text(test_code, encoding="utf-8")
        else:
            test_file_path.write_text("# Failed to generate tests", encoding="utf-8")
            
        # 5. Format with Black
        try:
            subprocess.run(["black", str(file_path), str(test_file_path)], capture_output=True)
        except Exception:
            pass # Ignore if black isn't installed
            
        return file_path, test_file_path, "success"

    def run_tests(self, test_file_path: Path) -> Tuple[bool, str]:
        """Run pytest on the generated test file."""
        try:
            # Add backend to pythonpath for the test run
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.backend_dir)
            result = subprocess.run(
                ["pytest", str(test_file_path), "-v"], 
                capture_output=True, text=True, env=env
            )
            return result.returncode == 0, result.stdout
        except Exception as e:
            return False, str(e)
