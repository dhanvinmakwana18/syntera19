from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
import shutil
import os
import uuid
import time
import base64
import requests
from core.config import settings

mm_router = APIRouter()

class MultiModalResponse(BaseModel):
    answer: str
    trace: list[dict]
    run_id: str

def query_vision_model(prompt: str, image_path: str):
    # Base64 encode the image
    with open(image_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
        
    payload = {
        "model": "llava",  # Default local vision model for Ollama
        "prompt": prompt,
        "images": [img_b64],
        "stream": False
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise Exception("Cannot connect to Ollama. Is it running?")
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            raise Exception("Vision model 'llava' not found. Please run `ollama pull llava`.")
        raise e
    except Exception as e:
        raise e

@mm_router.post("/chat/vision", response_model=MultiModalResponse)
async def vision_chat(query: str = Form(...), image: UploadFile = File(...)):
    start_time = time.time()
    run_id = str(uuid.uuid4())
    trace = []
    
    trace.append({"step": "Init", "action": f"Vision query received: '{query}'"})
    
    # Save image temporarily
    os.makedirs(os.path.join(settings.DATA_DIR, "temp"), exist_ok=True)
    temp_path = os.path.join(settings.DATA_DIR, "temp", f"{uuid.uuid4()}_{image.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
        
    trace.append({"step": "Preprocessing", "action": f"Image saved for processing: {image.filename}"})
    
    try:
        trace.append({"step": "Execution", "action": "Analyzing image via local Vision model (llava)"})
        answer = query_vision_model(query, temp_path)
        trace.append({"step": "Complete", "action": "Successfully generated vision response."})
    except Exception as e:
        answer = f"Error processing image: {str(e)}"
        trace.append({"step": "Error", "action": str(e)})
        
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    latency = round(time.time() - start_time, 2)
    trace.append({"step": "Timing", "action": f"Multi-modal response generated in {latency}s"})
    
    return MultiModalResponse(
        answer=answer,
        trace=trace,
        run_id=run_id
    )
