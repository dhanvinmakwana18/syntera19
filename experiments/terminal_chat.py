import os
import sys

# Fix for loading CUDA 12 DLLs bundled with PyTorch on Windows
import torch
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path):
    os.add_dll_directory(torch_lib_path)
    os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ.get("PATH", "")

from llama_cpp import Llama

# Locate the downloaded model
MODEL_FILE = os.path.join(os.path.dirname(__file__), "models", "Nemotron-Mini-4B-Instruct-Q4_K_M.gguf")

if not os.path.exists(MODEL_FILE):
    print(f"❌ Error: Model file not found at {MODEL_FILE}")
    print("Please ensure the download script ran successfully.")
    sys.exit(1)

print("🚀 Loading Nemotron-Mini-4B directly into RTX 4050 VRAM... (Please wait)")

# Initialize the model with stream and chat parameters
llm = Llama(
    model_path=MODEL_FILE,
    n_gpu_layers=-1,  # Full GPU Offload
    n_ctx=4096,       # Standard 4K context window
    verbose=False     # Disable C++ backend logs to keep the chat interface clean
)

print("\n" + "="*50)
print("🤖 NEMOTRON TERMINAL CHAT ACTIVE")
print("Hardware: NVIDIA GeForce RTX 4050 (CUDA Offload)")
print("Type 'quit' or 'exit' to end the session.")
print("="*50)

# The specific prompt format required by Nemotron-Mini-Instruct models
system_prompt = "You are a highly capable, smart, and concise AI assistant running entirely locally."
chat_history = f"<extra_id_0>System\n{system_prompt}\n"

while True:
    try:
        user_input = input("\n🧑 You: ")
        
        # Exit condition
        if user_input.strip().lower() in ['quit', 'exit']:
            print("\nShutting down local inference. Goodbye! 👋\n")
            break
            
        if not user_input.strip():
            continue
            
        # Append user input to the rolling conversation history
        chat_history += f"<extra_id_1>User\n{user_input}\n<extra_id_1>Assistant\n"
        
        print("🤖 Nemotron: ", end="", flush=True)
        
        response_text = ""
        
        # Stream the generation token-by-token
        stream = llm(
            chat_history,
            max_tokens=1024,
            stop=["<extra_id_1>", "<extra_id_0>"],
            stream=True,
            temperature=0.6,
            top_p=0.9
        )
        
        for chunk in stream:
            token = chunk['choices'][0]['text']
            print(token, end="", flush=True)
            response_text += token
            
        print() # Add a final newline when streaming completes
        
        # Append the assistant's response to the history so it remembers the conversation
        chat_history += f"{response_text}\n"
        
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user. Goodbye! 👋\n")
        break
    except Exception as e:
        print(f"\n❌ Error during generation: {str(e)}")
