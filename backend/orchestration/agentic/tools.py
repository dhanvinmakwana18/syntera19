from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel

class ToolResult(BaseModel):
    success: bool
    output: str
    metadata: Dict[str, Any] = {}

class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        pass
        
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

class RAGTool(BaseTool):
    def __init__(self, retrieval_pipeline, context_assembler):
        self.pipeline = retrieval_pipeline
        self.assembler = context_assembler

    @property
    def name(self) -> str:
        return "retrieve_documents"

    @property
    def description(self) -> str:
        return "Retrieves semantic documents from the knowledge base based on a query string."

    def execute(self, query: str = "", limit: int = 5, **kwargs) -> ToolResult:
        try:
            from core.domain import Query
            result = self.pipeline.run(Query(text=query), limit=limit)
            context = self.assembler.assemble(result.candidates)
            if not context.sources:
                return ToolResult(success=True, output="No documents found.")
            return ToolResult(success=True, output=context.text, metadata={"sources": context.sources})
        except Exception as e:
            return ToolResult(success=False, output=str(e))
