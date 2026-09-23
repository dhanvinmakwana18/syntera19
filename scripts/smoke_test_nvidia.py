import os
import sys

# Add backend to path so we can import Syntera modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from dotenv import load_dotenv
from core.container import build_container
from intelligence.contracts import IntelligenceRequest

def run_smoke_test():
    load_dotenv()
    if not os.environ.get("NVIDIA_API_KEY"):
        print("FAILURE: NVIDIA_API_KEY is missing from environment.")
        return

    print("Building application container...")
    container = build_container()
    
    print("Initializing intelligence core...")
    intelligence = container.get_intelligence()
    
    print("Testing NVIDIA Provider (Remote Nemotron)...")
    response = intelligence.generate(
        prompt="Write a 1-sentence summary of the benefits of CUDA.",
        temperature=0.3,
        max_tokens=100
    )
    
    if response.success:
        print("\nSUCCESS!")
        print(f"Model Identifier: {response.model}")
        print(f"Provider: {response.provider}")
        print(f"Latency: {response.latency_ms:.2f} ms")
        print(f"Response snippet: {response.content[:100]}...")
    else:
        print("\nFAILURE!")
        print(f"Model Identifier: {response.model}")
        print(f"Error: {response.error}")

if __name__ == "__main__":
    run_smoke_test()
