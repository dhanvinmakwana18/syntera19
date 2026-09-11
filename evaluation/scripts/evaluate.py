import requests
import time

def evaluate_retrieval():
    print("Evaluating Retrieval Latency and Precision...")
    
    # 1. Setup - we assume there are documents
    queries = [
        "What is machine learning?",
        "Explain the architecture",
        "Summarize the findings"
    ]
    
    total_latency = 0
    success = 0
    
    for q in queries:
        start = time.time()
        try:
            res = requests.post("http://localhost:8001/api/v1/chat", json={"query": q, "mode": "rag"}, timeout=120)
            if res.status_code == 200:
                success += 1
        except Exception as e:
            print(f"Failed query: {q} - {e}")
        latency = time.time() - start
        total_latency += latency
        
    avg = total_latency / len(queries)
    print(f"--- Evaluation Results ---")
    print(f"Queries Tested: {len(queries)}")
    print(f"Successful Executions: {success}/{len(queries)}")
    print(f"Average End-to-End Latency: {avg:.2f} seconds")
    
if __name__ == "__main__":
    evaluate_retrieval()
