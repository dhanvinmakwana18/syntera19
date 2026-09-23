from typing import List
from core.contracts import BaseQueryProcessor, BaseGenerator
from core.domain import Query, GenerationContext, GenerationResult

class PassthroughQueryProcessor(BaseQueryProcessor):
    def process(self, query: Query, **kwargs) -> List[Query]:
        return [query]

class StandardRAGGenerator(BaseGenerator):
    def __init__(self, intelligence_core):
        self.intelligence = intelligence_core
        
    def generate(self, query: Query, context: GenerationContext, **kwargs) -> GenerationResult:
        system_prompt = (
            "You are Syntera. Use the provided context to answer the user query.\n"
            "If the context does not contain the answer, say 'I cannot find the answer in the provided documents.'\n"
            "Always cite your sources using [Source X] notation. NEVER fabricate a source."
        )
        prompt = f"Context:\n{context.text}\n\nQuery: {query.text}"
        
        response = self.intelligence.generate(prompt=prompt, system_prompt=system_prompt)
        
        return GenerationResult(
            answer=response.content,
            raw_response=response.content,
            usage_metrics=response.usage
        )


def register(registry):
    registry.register_query_processor("passthrough", lambda **kwargs: PassthroughQueryProcessor())
    registry.register_generator("standard", lambda intelligence_core, **kwargs: StandardRAGGenerator(intelligence_core))
