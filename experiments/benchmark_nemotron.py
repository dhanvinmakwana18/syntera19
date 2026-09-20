import time
import os
import psutil
from huggingface_hub import hf_hub_download

# Fix for loading CUDA 12 DLLs bundled with PyTorch
import torch
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path):
    os.add_dll_directory(torch_lib_path)
    os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ.get("PATH", "")

from llama_cpp import Llama

MODEL_REPO = "bartowski/Nemotron-Mini-4B-Instruct-GGUF"
MODEL_FILE = "Nemotron-Mini-4B-Instruct-Q4_K_M.gguf"
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "models")

print("=== STEP 3: DOWNLOAD MODEL ===")
print(f"Downloading {MODEL_FILE} from {MODEL_REPO}...")
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=DOWNLOAD_DIR)
print(f"Model downloaded to: {model_path}")
print(f"File size: {os.path.getsize(model_path) / (1024**3):.2f} GB")

print("\n=== STEP 4 & 5: INFERENCE & GPU OFFLOAD VERIFICATION ===")
print("Loading model into llama.cpp with CUDA offload (n_gpu_layers=-1)...")

# Track RAM before load
process = psutil.Process(os.getpid())
ram_before = process.memory_info().rss / (1024**2)

t0 = time.time()
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1, # Offload all layers
    n_ctx=4096,      # Standard context
    verbose=True
)
load_time = time.time() - t0
ram_after = process.memory_info().rss / (1024**2)

print(f"\nModel Load Time: {load_time:.2f}s")
print(f"RAM Usage Increase: {ram_after - ram_before:.2f} MB")

prompt = "<extra_id_0>System\nYou are a helpful assistant.\n<extra_id_1>User\nExplain what a retrieval-augmented generation system does in three concise points.\n<extra_id_1>Assistant\n"

print("\n=== STEP 6 & 7: BENCHMARK & STABILITY ===")
num_runs = 3
results = []

for i in range(num_runs):
    print(f"\n--- Run {i+1} ---")
    t0 = time.time()
    
    # We use stream=False for simple benchmarking
    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.1,
        stop=["<extra_id_1>", "<extra_id_0>"]
    )
    
    t_end = time.time()
    
    # Extract stats from llama.cpp's native returned dict
    usage = output['usage']
    prompt_tokens = usage['prompt_tokens']
    completion_tokens = usage['completion_tokens']
    total_time = t_end - t0
    
    # Basic math
    gen_speed = completion_tokens / total_time
    
    print(f"Output: {output['choices'][0]['text'].strip()}")
    print(f"Prompt Tokens: {prompt_tokens}")
    print(f"Generated Tokens: {completion_tokens}")
    print(f"Generation Speed: {gen_speed:.2f} tokens/sec")
    
    results.append({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_time": total_time,
        "gen_speed": gen_speed
    })

print("\n=== FINAL BENCHMARK AVERAGES ===")
avg_gen_speed = sum(r['gen_speed'] for r in results) / num_runs
print(f"Average Generation Speed: {avg_gen_speed:.2f} tokens/sec")
