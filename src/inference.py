# anomalib_advanced/src/inference.py
import time
import pathlib
import numpy as np
from PIL import Image
from anomalib.deploy import OpenVINOInferencer
import cv2

class AdvancedInferencer:
    """
    Production-grade inference runner wrapper for Anomalib OpenVINO exports.
    Automatically handles formatting, type conversions, and latency tracking.
    """
    def __init__(self, model_xml_path: str, device: str = "AUTO"):
        self.path = pathlib.Path(model_xml_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"OpenVINO Model XML file not found at: {self.path}")
        
        print(f"Loading OpenVINO model from: {self.path} on device: {device}...")
        self.inferencer = OpenVINOInferencer(path=str(self.path), device=device)
        
    def predict(self, image_input) -> dict:
        """
        Run inference on a single image input.
        
        Args:
            image_input (str | Path | Image.Image | np.ndarray): 
                Image file path, PIL Image, or Numpy array.
                
        Returns:
            dict: Key-value parsed output dictionary:
                - anomaly_score (float)
                - pred_label (int)
                - anomaly_map (np.ndarray)
                - pred_mask (np.ndarray)
                - overlay (np.ndarray)
                - image (np.ndarray)
                - latency_ms (float)
        """
        image_path = None
        if isinstance(image_input, (str, pathlib.Path)):
            image_path = str(image_input)
            img_pil = Image.open(image_path).convert("RGB")
            img_np = np.array(img_pil)
        elif isinstance(image_input, Image.Image):
            img_pil = image_input.convert("RGB")
            img_np = np.array(img_pil)
        elif isinstance(image_input, np.ndarray):
            img_np = image_input
            img_pil = Image.fromarray(img_np)
        else:
            raise TypeError("Unsupported image input type. Provide path, PIL Image, or Numpy array.")

        start_time = time.perf_counter()
        
        # Run raw prediction (Anomalib OpenVINOInferencer will accept image path or numpy array)
        # Resize image to match model input size (256, 256)
        img_resized = cv2.resize(img_np, (256, 256))
        raw_pred = self.inferencer.predict(image=img_resized)
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Normalize prediction container if returned in a list
        if isinstance(raw_pred, (list, tuple)):
            if len(raw_pred) > 0:
                raw_pred = raw_pred[0]
            else:
                raise RuntimeError("Empty prediction list returned by inferencer.")

        # Robust helper to extract attributes or dictionary keys
        def get_field(obj, attr_name):
            if hasattr(obj, attr_name):
                return getattr(obj, attr_name)
            if isinstance(obj, dict):
                return obj.get(attr_name)
            return None

        # Convert Pytorch/Lightning tensors to Numpy arrays
        def to_numpy(arr):
            if arr is None:
                return None
            if hasattr(arr, "cpu"):
                arr = arr.cpu()
            if hasattr(arr, "numpy"):
                return arr.numpy()
            return np.array(arr)

        pred_score = get_field(raw_pred, "pred_score")
        pred_label = get_field(raw_pred, "pred_label")
        anomaly_map = get_field(raw_pred, "anomaly_map")
        pred_mask = get_field(raw_pred, "pred_mask")
        overlay = get_field(raw_pred, "segmentations")
        
        if overlay is None:
            overlay = get_field(raw_pred, "heat_map")  # Fallback to heatmap visual overlay
        def safe_squeeze(arr):
            if arr is None:
                return None
            return to_numpy(arr).squeeze()

        score_val = float(to_numpy(pred_score).item()) if pred_score is not None else 0.0
        label_val = int(to_numpy(pred_label).item()) if pred_label is not None else 0
        
        anomaly_map_np = safe_squeeze(anomaly_map)
        pred_mask_np = safe_squeeze(pred_mask)
        overlay_np = safe_squeeze(overlay)
        
        # Fallback overlay generation if OpenVINO didn't supply one
        if overlay_np is None:
            overlay_np = img_resized
            
        return {
            "anomaly_score": score_val,
            "pred_label": label_val,
            "anomaly_map": anomaly_map_np,
            "pred_mask": pred_mask_np,
            "overlay": overlay_np,
            "image": img_resized,
            "latency_ms": latency_ms
        }
