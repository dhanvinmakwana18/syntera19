import requests
import time
import os

BASE_URL = "http://localhost:8000/api/v1"

def wait_for_server():
    for _ in range(10):
        try:
            if requests.get(f"http://localhost:8000/health").status_code == 200:
                print("Server is up!")
                return True
        except:
            time.sleep(1)
    print("Server failed to start")
    return False

def run_tests():
    # 1. Create a dummy txt file
    with open("dummy_test.txt", "w", encoding="utf-8") as f:
        f.write("The quick brown fox jumps over the lazy dog. This document contains information about the secret code 4291.")
    
    # 2. Upload file
    print("Uploading file...")
    with open("dummy_test.txt", "rb") as f:
        res = requests.post(f"{BASE_URL}/kb/upload", files={"file": f})
    
    print("Upload response:", res.json())
    assert res.status_code == 200
    assert res.json()["chunks"] > 0
    
    # 3. Retrieve information
    print("Testing RAG retrieval...")
    res = requests.post(f"{BASE_URL}/chat", json={"query": "What is the secret code?", "mode": "rag"})
    data = res.json()
    print("Chat response:", data)
    assert res.status_code == 200
    
    # The source should contain dummy_test.txt
    sources = data.get("sources", [])
    assert any("dummy_test.txt" in s.get("filename", "") for s in sources)
    print("Retrieval test passed!")
    os.remove("dummy_test.txt")

if __name__ == "__main__":
    if wait_for_server():
        run_tests()
