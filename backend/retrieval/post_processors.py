from typing import List, Set
import uuid
from core.domain import RetrievalResult, Node
from core.contracts import BaseNodePostProcessor, BaseVectorStore

class NeighborExpansionPostProcessor(BaseNodePostProcessor):
    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store
        
    def process(self, nodes: List[RetrievalResult], expand_neighbors: bool = False, **kwargs) -> List[RetrievalResult]:
        if not expand_neighbors or not nodes:
            return nodes
            
        seen_texts = {n.node.text for n in nodes}
        existing_ids = {n.node.id for n in nodes}
        ids_to_fetch: Set[str] = set()
        
        for res in nodes:
            node = res.node
            source = node.metadata.get("source")
            chunk_index = node.metadata.get("chunk_index")
            
            if source and chunk_index is not None:
                prev_id = f"{source}_chunk_{chunk_index - 1}"
                next_id = f"{source}_chunk_{chunk_index + 1}"
                
                prev_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, prev_id))
                next_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, next_id))
                
                if prev_uuid not in existing_ids: 
                    ids_to_fetch.add(prev_uuid)
                if next_uuid not in existing_ids: 
                    ids_to_fetch.add(next_uuid)
                    
        if not ids_to_fetch:
            return nodes
            
        extra_nodes = self.vector_store.get_points_by_ids(list(ids_to_fetch))
        
        # Append extra nodes with a default score and mark them as expanded
        expanded_results = []
        for en in extra_nodes:
            if en.text and en.text not in seen_texts:
                seen_texts.add(en.text)
                en.metadata["is_expanded"] = True
                expanded_results.append(RetrievalResult(node=en, score=0.0))
                
        # Return combined list
        return nodes + expanded_results
# PHASE 4: Component Registration

def create_neighbor_expansion(vector_store=None) -> BaseNodePostProcessor:
    return NeighborExpansionPostProcessor(vector_store)




def register(registry):
    registry.register_post_processor('neighbor_expansion', create_neighbor_expansion)
