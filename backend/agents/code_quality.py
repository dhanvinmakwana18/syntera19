from enum import Enum
from typing import List, Dict
import ast
import os
from pathlib import Path

class Severity(Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

class CodeQualityAgent:
    def review(self, target_path: str, strictness: str = 'normal') -> dict:
        """
        Analyzes code for security problems, performance issues, and style smells.
        Returns a structured dictionary of issues.
        """
        target = Path(target_path)
        if not target.exists():
            return {'error': f'Target {target_path} not found'}
            
        # Simplified static analysis for MVP
        issues = []
        if target.is_file() and target.suffix == '.py':
            content = target.read_text(encoding='utf-8')
            if 'passwords=' in content or 'secret=' in content:
                issues.append({'type': 'security', 'message': 'Hardcoded secrets detected', 'severity': 'high'})
            if 'for ' in content and 'append' in content:
                issues.append({'type': 'performance', 'message': 'Inefficient loop detected (consider list comprehension)', 'severity': 'medium'})
                
        return {
            'language': 'python',
            'issues': issues,
            'refactoring': 'Consider extracting constants and using list comprehensions.',
            'severity': 'high' if any(i['severity'] == 'high' for i in issues) else 'low'
        }
