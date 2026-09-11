from test_semantic import semantic_chunk_text
with open('../data/documents/langchain_readme.txt', 'r', encoding='utf-8') as f:
    text = f.read()

c_sem = semantic_chunk_text(text, 1000, 200)
for i, c in enumerate(c_sem):
    print(f"--- Chunk {i} ({len(c)} chars) ---")
    print(c[:100].replace('\n', ' '))
    print("...")
