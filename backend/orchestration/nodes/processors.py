from typing import List
from core.contracts import BaseQueryProcessor, BaseGenerator
from core.domain import Query, GenerationContext, GenerationResult

class PassthroughQueryProcessor(BaseQueryProcessor):
    def process(self, query: Query, **kwargs) -> List[Query]:
        return [query]

class StandardRAGGenerator(BaseGenerator):
    def __init__(self, llm_provider):
        self.llm = llm_provider
        
    def generate(self, query: Query, context: GenerationContext, **kwargs) -> GenerationResult:
        system_prompt = (
            "You are Syntera. Use the provided context to answer the user query.\n"
            "If the context does not contain the answer, say 'I cannot find the answer in the provided documents.'\n"
            "Always cite your sources using [Source X] notation. NEVER fabricate a source."
        )
        prompt = f"Context:\n{context.text}\n\nQuery: {query.text}"
        
        raw_response = self.llm.generate(prompt=prompt, system_prompt=system_prompt)
        
        return GenerationResult(
            answer=raw_response,
            raw_response=raw_response,
            usage_metrics={}
        )


def register(registry):
    registry.register_query_processor("passthrough", lambda **kwargs: PassthroughQueryProcessor())
    registry.register_generator("standard", lambda llm_provider, **kwargs: StandardRAGGenerator(llm_provider))
