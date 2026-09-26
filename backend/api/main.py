import os
import uuid
import time
from typing import List
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
import sqlite3

# Import our agent
from backend.agents.data_scientist import DataScientistAgent

app = FastAPI(title="Syntera API", version="0.2.0")

# --- In-Memory Job Queue (MVP) ---
JOBS = {}

# --- Monitoring & Metrics (Prometheus Format + SQLite) ---
DB_PATH = Path("artifacts/metrics.db")
DB_PATH.parent.mkdir(exist_ok=True)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS metrics 
                     (id INTEGER PRIMARY KEY, endpoint TEXT, method TEXT, 
                     latency REAL, status INTEGER, timestamp REAL)''')
init_db()

# Simple Prometheus Counters
class Metrics:
    requests_total = 0
    errors_total = 0
    total_latency = 0.0

@app.middleware("http")
async def monitor_requests(request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        Metrics.errors_total += 1
        raise e
    finally:
        latency = time.time() - start_time
        Metrics.requests_total += 1
        Metrics.total_latency += latency
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO metrics (endpoint, method, latency, status, timestamp) VALUES (?, ?, ?, ?, ?)",
                         (request.url.path, request.method, latency, status_code, time.time()))
    return response

@app.get("/metrics", response_class=PlainTextResponse)
def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    return f"""# HELP syntera_requests_total Total API requests
# TYPE syntera_requests_total counter
syntera_requests_total {Metrics.requests_total}
# HELP syntera_errors_total Total API errors
# TYPE syntera_errors_total counter
syntera_errors_total {Metrics.errors_total}
# HELP syntera_latency_seconds_total Total latency
# TYPE syntera_latency_seconds_total counter
syntera_latency_seconds_total {Metrics.total_latency}
"""

# --- Job Processing ---
def run_analysis_task(job_id: str, file_paths: List[str], goal: str):
    JOBS[job_id]["status"] = "processing"
    try:
        agent = DataScientistAgent()
        result = agent.analyze(file_paths, goal)
        if "error" in result:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = result["error"]
        else:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["result"] = result
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)

# --- Endpoints ---
@app.post("/analyze")
async def analyze_datasets(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...), goal: str = "Perform comprehensive EDA and model training."):
    job_id = str(uuid.uuid4())
    temp_dir = Path(f"artifacts/temp_{job_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    file_paths = []
    for f in files:
        file_path = temp_dir / f.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await f.read())
        file_paths.append(str(file_path))
        
    JOBS[job_id] = {"status": "queued", "files": [f.filename for f in files], "created_at": time.time()}
    background_tasks.add_task(run_analysis_task, job_id, file_paths, goal)
    return {"job_id": job_id, "status": "queued", "message": "Background analysis started."}

@app.get("/status/{job_id}")
def check_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]

@app.get("/report/{job_id}")
def get_report(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Report not ready yet")
        
    pdf_path = job["result"].get("report_pdf")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF report file not found on disk")
        
    return FileResponse(path=pdf_path, filename=f"syntera_report_{job_id}.pdf", media_type="application/pdf")

@app.delete("/report/{job_id}")
def delete_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS.pop(job_id)
    # Cleanup temp files
    temp_dir = Path(f"artifacts/temp_{job_id}")
    if temp_dir.exists():
        for f in temp_dir.iterdir():
            f.unlink()
        temp_dir.rmdir()
    return {"status": "deleted", "job_id": job_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
