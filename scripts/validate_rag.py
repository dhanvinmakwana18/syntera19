import sys
import os
import json

sys.path.insert(0, os.path.abspath('syntera/backend'))

from retrieval.pipeline import retrieve_documents

print("\n--- VALIDATING STANDARD RETRIEVAL (No Expansion) ---")
context, sources = retrieve_documents("What is a Document in LangChain?", limit=5, expand_neighbors=False)
print("Context Output:\n", context[:500], "...\n")
print("Sources JSON:\n", json.dumps(sources, indent=2))

print("\n--- VALIDATING EXPANDED RETRIEVAL ---")
context_exp, sources_exp = retrieve_documents("What is a Document in LangChain?", limit=5, expand_neighbors=True)
print("Expanded Context Output:\n", context_exp[:500], "...\n")
print("Expanded Sources JSON:\n", json.dumps(sources_exp, indent=2))
