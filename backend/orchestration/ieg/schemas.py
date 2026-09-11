from pydantic import BaseModel, Field
from typing import List, Optional

class SubQuery(BaseModel):
    id: str = Field(..., description="Unique identifier for this subquery (e.g., q1, q2).")
    query: str = Field(..., description="The specific search query to execute against the vector database.")
    purpose: str = Field(..., description="Why this information is needed to answer the original question.")
    dependency: Optional[str] = Field(None, description="If this query depends on the answer to another subquery, list its ID here.")

class QueryDecomposition(BaseModel):
    subqueries: List[SubQuery] = Field(..., description="The list of decomposed subqueries.")

class EvidenceEvaluation(BaseModel):
    sufficient: bool = Field(..., description="True if the currently collected evidence is completely sufficient to answer the original query.")
    reasoning: str = Field(..., description="Detailed explanation of why the evidence is sufficient or insufficient.")
    missing_information: List[str] = Field(default_factory=list, description="Specific facts or topics that are still missing. Empty if sufficient.")
    follow_up_queries: List[SubQuery] = Field(default_factory=list, description="Targeted new subqueries to find the missing information. Empty if sufficient.")
