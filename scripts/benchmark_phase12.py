import os
import sys
import time
import psutil
import requests
from dotenv import load_dotenv

# Load root .env
root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(root_env)

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from core.container import build_container
from intelligence.contracts import IntelligenceRequest, TaskComplexity

def print_header(title):
    print(f"\n{'='*50}\n{title}\n{'='*50}")

def get_vram():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / (1024**3), torch.cuda.memory_reserved(0) / (1024**3)
    except:
        pass
    return 0.0, 0.0

def benchmark_local():
    print_header("LOCAL INFERENCE BENCHMARK")
    print("Checking if local llama.cpp server is running on :8080...")
    try:
        r = requests.get("http://localhost:8080/v1/models", timeout=2)
        if r.status_code == 200:
            print("Local server is ONLINE.")
            # We will perform a local call
            start = time.time()
            res = requests.post("http://localhost:8080/v1/chat/completions", json={
                "model": "Nemotron-Mini-4B-Instruct-Q4_K_M",
                "messages": [{"role": "user", "content": "Hello, write a short poem."}],
                "max_tokens": 50
            }, timeout=30)
            elapsed = time.time() - start
            if res.status_code == 200:
                data = res.json()
                content = data['choices'][0]['message']['content']
                tokens = data['usage']['completion_tokens']
                print(f"Success! Generated {tokens} tokens in {elapsed:.2f}s ({tokens/elapsed:.2f} tok/s)")
            else:
                print("Failed generation:", res.status_code)
        else:
            print("Server returned:", r.status_code)
    except Exception as e:
        print("Local server is OFFLINE or unreachable:", e)
        print("Skipping active local benchmark.")

def benchmark_remote(container):
    print_header("REMOTE INFERENCE BENCHMARK")
    intel = container.get_intelligence()
    
    start = time.time()
    try:
        res = intel.generate(
            prompt="Explain the theory of relativity in 2 short sentences.",
            complexity=TaskComplexity.COMPLEX
        )
        elapsed = time.time() - start
        
        if res.success:
            print(f"Success! Model: {res.model}, Provider: {res.provider}")
            print(f"Latency: {res.latency_ms:.2f}ms")
            print(f"Response: {res.content}")
        else:
            print("Failed generation:", res.error)
    except Exception as e:
        print("Exception during remote benchmark:", e)

def benchmark_router(container):
    print_header("ROUTER DETERMINISM BENCHMARK")
    intel = container.get_intelligence()
    router = intel.router
    
    tasks = [
        ("Simple Chat", IntelligenceRequest(prompt="Hello", complexity=TaskComplexity.SIMPLE)),
        ("Heavy Reasoning", IntelligenceRequest(prompt="Solve P=NP", complexity=TaskComplexity.COMPLEX)),
        ("Agentic Task", IntelligenceRequest(prompt="Use a tool to find X", task="agent")),
        ("Multimodal Task", IntelligenceRequest(prompt="Analyze this image", task="multimodal")),
    ]
    
    for name, req in tasks:
        provider = router.route(req)
        print(f"{name:<20} -> {provider.profile.name} (Local: {provider.profile.local}, Tier: {provider.profile.cost_tier})")

def benchmark_rag(container):
    print_header("RAG PIPELINE INTEGRATION")
    print("Checking if NVIDIAEmbeddingProvider is registered...")
    emb = container.get_embedding_provider()
    print(f"Embedding provider class: {emb.__class__.__name__}")
    if hasattr(emb, "model_name"):
        print(f"Model: {emb.model_name}")
        try:
            vec = emb.embed_text("Test embedding")
            print(f"Successfully generated embedding of size {len(vec)}")
        except Exception as e:
            print(f"Embedding failed: {e}")
            
    print("\nChecking Reranker...")
    rerank = container.get_reranker()
    print(f"Reranker class: {rerank.__class__.__name__}")
    if hasattr(rerank, "model_name"):
        print(f"Model: {rerank.model_name}")

def main():
    print("Initializing Application Container...")
    container = build_container()
    
    vram_alloc, vram_res = get_vram()
    mem = psutil.virtual_memory()
    print(f"System RAM: {mem.used/(1024**3):.2f}GB / {mem.total/(1024**3):.2f}GB")
    print(f"System VRAM: {vram_alloc:.2f}GB allocated, {vram_res:.2f}GB reserved")
    
    benchmark_local()
    benchmark_remote(container)
    benchmark_router(container)
    benchmark_rag(container)
    print_header("BENCHMARK COMPLETE")

if __name__ == '__main__':
    main()
