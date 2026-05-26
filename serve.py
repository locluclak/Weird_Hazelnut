import io
import os
import pathlib
import sys
import time
from typing import Optional

import numpy as np
import uvicorn
import mlflow
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

# Add current workspace directory to python path for modular imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference import AdvancedInferencer

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "results/exported/weights/openvino/model.xml")
DEVICE = os.getenv("INFERENCE_DEVICE", "CPU")
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")

# MLflow Configuration
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "anomalib_production_api")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

from contextlib import asynccontextmanager

# Global instances
inferencer: Optional[AdvancedInferencer] = None
mlflow_run = None
request_count = 0
request_metrics = {
    "anomaly_score": [],
    "latency_ms": [],
    "pred_label": []
}


def _log_mlflow_figure(name: str, fig) -> None:
    """Log a matplotlib figure to MLflow if the run is active."""
    if not mlflow_run:
        return
    try:
        mlflow.log_figure(fig, name)
    except Exception as e:
        print(f"[WARNING] Failed to log MLflow artifact '{name}': {e}")
    finally:
        try:
            fig.clf()
        except Exception:
            pass


def log_mlflow_histograms(force: bool = False) -> None:
    """Create and log histogram artifacts for collected request metrics."""
    if not mlflow_run or len(request_metrics["anomaly_score"]) == 0:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARNING] matplotlib import failed; histogram artifacts will not be generated: {e}")
        return

    try:
        # Always refresh the artifact images so they are visible during a live run.
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(request_metrics["anomaly_score"], bins=20, range=(0, 1), color="#1f77b4", edgecolor="black")
        ax.set_title("Anomaly Score Distribution")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        _log_mlflow_figure("histogram_anomaly_score.png", fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(request_metrics["latency_ms"], bins=20, color="#2ca02c", edgecolor="black")
        ax.set_title("Inference Latency Distribution")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        _log_mlflow_figure("histogram_latency_ms.png", fig)
        plt.close(fig)

        labels = request_metrics["pred_label"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(labels, bins=[-0.5, 0.5, 1.5], color="#c44e52", edgecolor="black")
        ax.set_title("Prediction Label Distribution")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["GOOD", "ANOMALY"])
        fig.tight_layout()
        _log_mlflow_figure("histogram_pred_label.png", fig)
        plt.close(fig)
    except Exception as e:
        print(f"[WARNING] Failed to generate MLflow histogram artifacts: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the OpenVINO model and initialize MLflow tracking."""
    global inferencer, mlflow_run
    
    # 1. Load Model
    model_xml = pathlib.Path(MODEL_PATH).resolve()
    if not model_xml.exists():
        print(f"[ERROR] Model not found at {model_xml}. API will fail to start.")
    else:
        try:
            inferencer = AdvancedInferencer(model_xml_path=str(model_xml), device=DEVICE)
            print(f"Successfully loaded model from {model_xml} on {DEVICE}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")

    # 2. Initialize MLflow
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        
        # Enable system metrics logging (CPU, Memory, Disk, etc.)
        # Requires 'psutil' which is already in our environment
        mlflow.enable_system_metrics_logging()
        
        # Start a persistent run for this service session
        mlflow_run = mlflow.start_run(run_name=f"api_session_{int(time.time())}")
        mlflow.log_params({
            "model_path": MODEL_PATH,
            "device": DEVICE,
            "api_host": HOST,
            "api_port": PORT,
            "system_metrics": "enabled"
        })
        print(f"MLflow tracking initialized: {MLFLOW_TRACKING_URI} (Exp: {MLFLOW_EXPERIMENT})")
    except Exception as e:
        print(f"[WARNING] MLflow initialization failed: {e}. Inference will continue without logging.")

    yield
    
    # Cleanup
    if mlflow_run:
        log_mlflow_histograms()
        mlflow.end_run()
        print("MLflow run ended.")
    print("Shutting down API...")

app = FastAPI(
    title="Anomalib Production API",
    description="High-performance anomaly detection service using OpenVINO with MLflow tracking.",
    version="1.1.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy", 
        "model": MODEL_PATH, 
        "device": DEVICE,
        "mlflow_tracking": True if mlflow_run else False,
        "total_requests": request_count
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Perform anomaly detection on an uploaded image and log to MLflow.
    """
    global request_count
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model not loaded or failed to initialize.")

    # Validate file type
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        ext = pathlib.Path(file.filename).suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            raise HTTPException(status_code=400, detail=f"File must be an image. Got: {content_type}")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run inference
        results = inferencer.predict(image)
        request_count += 1
        request_metrics["anomaly_score"].append(float(results["anomaly_score"]))
        request_metrics["latency_ms"].append(float(results["latency_ms"]))
        request_metrics["pred_label"].append(int(results["pred_label"]))

        # MLflow Logging (Async/Fire-and-forget logic could be added here for higher concurrency)
        if mlflow_run:
            try:
                # Log metrics for this specific request using step for time-series visualization.
                mlflow.log_metrics({
                    "inference_score": float(results["anomaly_score"]),
                    "inference_label": int(results["pred_label"]),
                    "latency_ms": float(results["latency_ms"]),
                    "total_requests": request_count
                }, step=request_count)
                # Log histogram artifacts as images so the score and label distributions are visible.
                log_mlflow_histograms(force=True)
            except Exception as log_e:
                print(f"[WARNING] Failed to log to MLflow: {log_e}")

        # Format response
        response = {
            "filename": file.filename,
            "anomaly_score": float(results["anomaly_score"]),
            "pred_label": int(results["pred_label"]),
            "label_name": "ANOMALY" if results["pred_label"] == 1 else "GOOD",
            "latency_ms": float(results["latency_ms"]),
        }
        
        return JSONResponse(content=response)

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
