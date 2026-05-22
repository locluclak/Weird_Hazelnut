# anomalib_advanced/predict.py
import argparse
import os
import pathlib
import sys
import time
import cv2
import mlflow
import numpy as np
from PIL import Image
import traceback
# Add current workspace directory to python path for modular imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import AdvancedInferencer
from src.utils import create_overlay_collage

def get_images(input_path: pathlib.Path) -> list[pathlib.Path]:
    """Gather all image paths from file or directory path."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    if input_path.is_file():
        if input_path.suffix.lower() in exts:
            return [input_path]
        return []
    
    images = []
    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            images.append(path)
    return images

def main(args):
    model_path = pathlib.Path(args.model).resolve()
    image_input = pathlib.Path(args.image).resolve()
    output_dir = pathlib.Path(args.output).resolve()
    
    # 1. Gather all files to predict
    if not image_input.exists():
        print(f"Input image path does not exist: {image_input}")
        sys.exit(1)
        
    image_paths = get_images(image_input)
    if not image_paths:
        print(f"No valid image files found at: {image_input}")
        sys.exit(1)
        
    print(f"Found {len(image_paths)} images for prediction.")
    
    # 2. Load model
    if not model_path.exists():
        print(f"Model path XML file not found at: {model_path}")
        sys.exit(1)
        
    # Check if there is an accompanying bin file (must be there for OpenVINO)
    bin_path = model_path.with_suffix(".bin")
    if not bin_path.exists():
        print(f"[WARNING] Matching weight bin file not found at: {bin_path}. OpenVINO execution may fail if weights are not in the same directory.")
        
    inferencer = AdvancedInferencer(model_xml_path=str(model_path), device=args.device)
    
    # 3. Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Handle MLflow context if requested
    mlflow_enabled = args.mlflow_log
    if mlflow_enabled:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("Production_Inference")
        
        # Start a dedicated production batch inference run
        run_name = f"Batch_Inference_{time.strftime('%Y%m%d_%H%M%S')}"
        mlflow_run = mlflow.start_run(run_name=run_name)
        print(f"\nStarted MLflow Logging context under run: '{run_name}'")
        mlflow.log_param("model_used", model_path.name)
        mlflow.log_param("device", args.device)
        mlflow.log_param("total_images_processed", len(image_paths))
        
    print("\n--- Running Predictions ---")
    
    latencies = []
    
    try:
        for idx, img_path in enumerate(image_paths):
            print(f"\nProcessing [{idx+1}/{len(image_paths)}]: {img_path.name}")
            
            # Predict
            pred = inferencer.predict(str(img_path))
            latency = pred["latency_ms"]
            latencies.append(latency)
            
            score = pred["anomaly_score"]
            pred_label = pred["pred_label"]
            pred_label_str = "ANOMALY" if pred_label == 1 else "GOOD"
            
            print(f"  Result: {pred_label_str} | Anomaly Score: {score:.4f} | Latency: {latency:.2f}ms")
            
            # Generate custom overlay collage
            img_np = pred["image"]
            heatmap_norm = (pred["anomaly_map"] - pred["anomaly_map"].min()) / (pred["anomaly_map"].max() - pred["anomaly_map"].min() + 1e-8)
            mask_np = pred["pred_mask"]
            
            # Create a nice overlay blend
            gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            gray_img_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
            heatmap_color = cv2.applyColorMap((heatmap_norm.squeeze() * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            h, w = gray_img_rgb.shape[:2]

            # 2. Resize the heatmap to perfectly match the original image size
            # Note: cv2.resize expects (Width, Height) order for the size argument
            heatmap_resized = cv2.resize(heatmap_color_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            overlay_np = cv2.addWeighted(gray_img_rgb, 0.65, heatmap_resized, 0.35, 0)
            
            # Guess ground truth from filename if it exists in a folder named test/anomaly or good
            # Or just mark it 0 if it's normal (good) and 1 otherwise
            gt_label = 0
            if "good" in str(img_path.parent).lower():
                gt_label = 0
            elif any(cat in str(img_path.parent).lower() for cat in ["crack", "cut", "hole", "print"]):
                gt_label = 1
                
            out_collage_name = f"pred_{img_path.stem}.png"
            out_collage_path = output_dir / out_collage_name
            
            # Build and save collage using utils helper
            create_overlay_collage(
                image=img_np,
                heatmap=heatmap_norm,
                mask=mask_np,
                overlay=overlay_np,
                gt_label=gt_label,
                pred_label=pred_label,
                score=score,
                threshold=args.threshold,
                output_path=str(out_collage_path)
            )
            print(f"  Collage visual output saved to: {out_collage_path}")
            
            # Log individual predictions to MLflow if enabled
            if mlflow_enabled:
                # Log score and latency
                mlflow.log_metric(f"score_{img_path.stem}", score)
                mlflow.log_metric(f"latency_{img_path.stem}", latency)
                
                # Upload the visual result collage
                mlflow.log_image(
                    Image.open(str(out_collage_path)),
                    f"predictions/pred_{img_path.stem}_score_{score:.3f}.png"
                )
                
        # Run summaries
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = np.percentile(latencies, 95)
        print(f"\n==============================================================================")
        print("Batch prediction completed successfully!")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  95th-percentile: {p95_latency:.2f}ms")
        print(f"==============================================================================")
        
        if mlflow_enabled:
            mlflow.log_metric("avg_latency_ms", float(avg_latency))
            mlflow.log_metric("p95_latency_ms", float(p95_latency))
            mlflow.end_run()
            print("Successfully uploaded inference logs and dashboard collages to MLflow.")
            
    except Exception as e:
        print(f"[ERROR] Inference run aborted: {e}")
        traceback.print_exc()  
        
        if mlflow_enabled:
            mlflow.end_run()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Anomalib production inference CLI.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the exported OpenVINO model.xml file."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to a single hazelnut image or folder of images to process."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/predictions",
        help="Directory to save visual overlays and dashboard diagrams."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Anomaly decision threshold (used for labeling if model threshold isn't baked in)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="AUTO",
        help="OpenVINO inference device target (AUTO, CPU, GPU)."
    )
    parser.add_argument(
        "--mlflow-log",
        action="store_true",
        help="Enable direct visual prediction tracking and latency uploads to MLflow."
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="MLflow tracking server URI endpoint."
    )
    args = parser.parse_args()
    main(args)
