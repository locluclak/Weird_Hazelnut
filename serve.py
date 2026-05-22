# anomalib_advanced/serve.py
import io
import os
import pathlib
import sys
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

# Add current workspace directory to python path for modular imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference import AdvancedInferencer

# Configuration (could be moved to a .env or config file)
MODEL_PATH = os.getenv("MODEL_PATH", "results/exported/weights/openvino/model.xml")
DEVICE = os.getenv("INFERENCE_DEVICE", "CPU")
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")

from contextlib import asynccontextmanager

# Global inferencer instance
inferencer: Optional[AdvancedInferencer] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the OpenVINO model into memory at startup."""
    global inferencer
    model_xml = pathlib.Path(MODEL_PATH).resolve()
    if not model_xml.exists():
        print(f"[ERROR] Model not found at {model_xml}. API will fail to start.")
    else:
        try:
            inferencer = AdvancedInferencer(model_xml_path=str(model_xml), device=DEVICE)
            print(f"Successfully loaded model from {model_xml} on {DEVICE}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
    yield
    # Cleanup if needed
    print("Shutting down API...")

app = FastAPI(
    title="Anomalib Production API",
    description="High-performance anomaly detection service using OpenVINO.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": MODEL_PATH, "device": DEVICE}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Perform anomaly detection on an uploaded image.
    
    Returns:
        JSON response with anomaly score, label, and latency.
    """
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model not loaded or failed to initialize.")

    # Validate file type (handle None content_type)
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        # Fallback: check extension
        ext = pathlib.Path(file.filename).suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            raise HTTPException(status_code=400, detail=f"File must be an image. Got: {content_type}")

    try:
        # Read image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run inference
        # AdvancedInferencer.predict handles the resizing to 256x256 internally
        results = inferencer.predict(image)
        
        # Format response
        # We exclude the large numpy arrays (image, anomaly_map, overlay) 
        # from the default JSON response for efficiency.
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
