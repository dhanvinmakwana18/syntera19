import pytest
from backend.agents.code_quality import (
    Severity,
    IssueCategory,
    CodeIssue,
    RefactoringSuggestion,
    AnalysisResult,
)


class TestSeverity:
    """Tests for Severity enum."""

    def test_severity_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFO.value == "info"

    def test_severity_ordering(self):
        assert Severity.INFO < Severity.LOW
        assert Severity.LOW < Severity.MEDIUM
        assert Severity.MEDIUM < Severity.HIGH
        assert Severity.HIGH < Severity.CRITICAL
        assert Severity.INFO < Severity.CRITICAL

    def test_severity_equality(self):
        assert Severity.HIGH == Severity.HIGH
        assert Severity.HIGH != Severity.MEDIUM


class TestIssueCategory:
    """Tests for IssueCategory enum."""

    def test_category_values(self):
        assert IssueCategory.SECURITY.value == "security"
        assert IssueCategory.PERFORMANCE.value == "performance"
        assert IssueCategory.STYLE.value == "style"
        assert IssueCategory.MAINTAINABILITY.value == "maintainability"
        assert IssueCategory.BUG_RISK.value == "bug_risk"


class TestCodeIssue:
    """Tests for CodeIssue dataclass."""

    def test_code_issue_creation_minimal(self):
        issue = CodeIssue(
            id="test-1",
            category=IssueCategory.SECURITY,
            severity=Severity.HIGH,
            message="Test issue",
            file_path="test.py",
            line_number=10,
        )
        assert issue.id == "test-1"
        assert issue.category == IssueCategory.SECURITY
        assert issue.severity == Severity.HIGH
        assert issue.message == "Test issue"
        assert issue.file_path == "test.py"
        assert issue.line_number == 10
        assert issue.column == 0
        assert issue.code_snippet == ""
        assert issue.rule_id == ""
        assert issue.confidence == 1.0
        assert issue.refactoring_suggestion is None
        assert issue.metadata == {}

    def test_code_issue_creation_full(self):
        issue = CodeIssue(
            id="test-2",
            category=IssueCategory.PERFORMANCE,
            severity=Severity.MEDIUM,
            message="Performance issue",
            file_path="perf.py",
            line_number=25,
            column=5,
            code_snippet="slow_code()",
            rule_id="PERF001",
            confidence=0.9,
            refactoring_suggestion="Use faster algorithm",
            metadata={"key": "value"},
        )
        assert issue.column == 5
        assert issue.code_snippet == "slow_code()"
        assert issue.rule_id == "PERF001"
        assert issue.confidence == 0.9
        assert issue.refactoring_suggestion == "Use faster algorithm"
        assert issue.metadata == {"key": "value"}

    def test_code_issue_to_dict(self):
        issue = CodeIssue(
            id="test-3",
            category=IssueCategory.STYLE,
            severity=Severity.LOW,
            message="Style issue",
            file_path="style.py",
            line_number=5,
            column=2,
            code_snippet="bad_style",
            rule_id="STYLE001",
            confidence=0.8,
            refactoring_suggestion="Fix style",
            metadata={"detail": "info"},
        )
        d = issue.to_dict()
        assert d["id"] == "test-3"
        assert d["category"] == "style"
        assert d["severity"] == "low"
        assert d["message"] == "Style issue"
        assert d["file_path"] == "style.py"
        assert d["line_number"] == 5
        assert d["column"] == 2
        assert d["code_snippet"] == "bad_style"
        assert d["rule_id"] == "STYLE001"
        assert d["confidence"] == 0.8
        assert d["refactoring_suggestion"] == "Fix style"
        assert d["metadata"] == {"detail": "info"}


class TestRefactoringSuggestion:
    """Tests for RefactoringSuggestion dataclass."""

    def test_refactoring_suggestion_creation(self):
        suggestion = RefactoringSuggestion(
            id="ref-1",
            issue_id="issue-1",
            title="Extract Method",
            description="Extract duplicated code into a method",
            original_code="def foo():\n    x = 1\n    y = 2",
            suggested_code="def foo():\n    bar()\n\ndef bar():\n    x = 1\n    y = 2",
            file_path="refactor.py",
            line_start=10,
            line_end=12,
            estimated_effort="medium",
            impact="high",
        )
        assert suggestion.id == "ref-1"
        assert suggestion.issue_id == "issue-1"
        assert suggestion.title == "Extract Method"
        assert suggestion.description == "Extract duplicated code into a method"
        assert suggestion.original_code == "def foo():\n    x = 1\n    y = 2"
        assert suggestion.suggested_code == "def foo():\n    bar()\n\ndef bar():\n    x = 1\n    y = 2"
        assert suggestion.file_path == "refactor.py"
        assert suggestion.line_start == 10
        assert suggestion.line_end == 12
        assert suggestion.estimated_effort == "medium"
        assert suggestion.impact == "high"

    def test_refactoring_suggestion_to_dict(self):
        suggestion = RefactoringSuggestion(
            id="ref-2",
            issue_id="issue-2",
            title="Inline Variable",
            description="Inline the variable",
            original_code="x = 5\nprint(x)",
            suggested_code="print(5)",
            file_path="inline.py",
            line_start=1,
            line_end=2,
            estimated_effort="low",
            impact="low",
        )
        d = suggestion.to_dict()
        assert d["id"] == "ref-2"
        assert d["issue_id"] == "issue-2"
        assert d["title"] == "Inline Variable"
        assert d["description"] == "Inline the variable"
        assert d["original_code"] == "x = 5\nprint(x)"
        assert d["suggested_code"] == "print(5)"
        assert d["file_path"] == "inline.py"
        assert d["line_start"] == 1
        assert d["line_end"] == 2
        assert d["estimated_effort"] == "low"
        assert d["impact"] == "low"


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_analysis_result_creation_minimal(self):
        result = AnalysisResult(
            language="python",
            issues=[],
            refactoring_suggestions=[],
            overall_severity=Severity.INFO,
        )
        assert result.language == "python"
        assert result.issues == []
        assert result.refactoring_suggestions == []
        assert result.overall_severity == Severity.INFO
        assert result.summary == {}
        assert result.files_analyzed == 0

    def test_analysis_result_creation_full(self):
        issue = CodeIssue(
            id="issue-1",
            category=IssueCategory.SECURITY,
            severity=Severity.HIGH,
            message="Security issue",
            file_path="sec.py",
