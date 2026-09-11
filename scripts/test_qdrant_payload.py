import sys
import os
sys.path.insert(0, os.path.abspath('syntera/backend'))

from vectorstore.qdrant_client import vector_store
res = vector_store.search("LangChain", limit=1)
print(res[0]['payload'])
