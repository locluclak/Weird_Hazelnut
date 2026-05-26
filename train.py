# anomalib_advanced/train.py
import argparse
import os
import pathlib
import sys
import yaml
import numpy as np
import cv2
import mlflow
from PIL import Image

# Add current workspace directory to python path for modular imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomalib.loggers import AnomalibMLFlowLogger
from src.data import detect_data_leakage, get_dataset_stats
from src.engine import run_training

def main(args):
    # 1. Load configuration file
    config_path = pathlib.Path(args.config).resolve()
    if not config_path.exists():
        print(f"Configuration file not found at: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print("--- Loaded Configuration Settings ---")
    print(yaml.dump(config, default_flow_style=False))
    
    # 2. Setup MLflow Tracking Details
    mlflow_config = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])
    
    # Ensure local directory for temp results exists
    os.makedirs("./results", exist_ok=True)
    
    # 3. Start manual MLflow context
    print("\n==============================================================================")
    print(f"Starting MLflow Run: {mlflow_config['run_name']}")
    print("==============================================================================")
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as run:
        run_id = run.info.run_id
        print(f"Active MLflow Run ID: {run_id}")
        
        # A. Execute Data Leakage Detection
        print("\n--- Running Data Leakage Checker ---")
        
        # Robustly resolve hazelnut dataset root relative to CWD, config, or workspace root
        data_root = config["data"]["root"]
        data_root_path = pathlib.Path(data_root)
        if not data_root_path.exists():
            workspace_root = pathlib.Path(__file__).resolve().parent.parent
            if (workspace_root / data_root).exists():
                data_root_path = workspace_root / data_root
            elif (workspace_root / "hazelnut").exists():
                data_root_path = workspace_root / "hazelnut"
                
        config["data"]["root"] = str(data_root_path.resolve())
        print(f"Resolved Dataset Root Path: {config['data']['root']}")
        
        duplicates = detect_data_leakage(config["data"]["root"])
        
        mlflow.log_param("data_leakage_detected", len(duplicates) > 0)
        mlflow.log_param("data_leakage_count", len(duplicates))
        
        if duplicates:
            print(f"[WARNING] Data leakage detected! Found {len(duplicates)} duplicates between train & test sets.")
            for i, (train_img, test_img) in enumerate(duplicates[:5]):
                print(f"  Leakage {i}: train={train_img.name} <=> test={test_img.name}")
        else:
            print("[SUCCESS] No data leakage detected! Dataset is split correctly.")
            
        # B. Gather Dataset Statistics
        stats = get_dataset_stats(config["data"]["root"])
        print(f"Dataset summary counts: {stats}")
        for path_name, count in stats.items():
            mlflow.log_param(f"dataset_count_{path_name.replace('/', '_')}", count)
            
        # C. Log Config Hyperparameters to MLflow
        mlflow.log_params({
            "model_name": config["model"]["name"],
            "model_backbone": config["model"]["backbone"],
            "train_batch_size": config["data"].get("train_batch_size", 32),
            "engine_max_epochs": config["engine"].get("max_epochs", 1),
            "export_type": config["export"].get("export_type", "OPENVINO")
        })
        
        if config["model"]["name"].lower() == "patchcore":
            mlflow.log_params({
                "patchcore_coreset_ratio": config["model"].get("coreset_sampling_ratio", 0.01),
                "patchcore_neighbors": config["model"].get("num_neighbors", 9)
            })
        elif config["model"]["name"].lower() == "padim":
            mlflow.log_param("padim_layers", str(config["model"].get("layers")))
            
        # D. Set up the Anomalib Logger using the exact active run
        mlflow_logger = AnomalibMLFlowLogger(
            tracking_uri=mlflow_config["tracking_uri"],
            experiment_name=mlflow_config["experiment_name"],
            run_id=run_id
        )
        
        # E. Invoke training pipeline
        model, datamodule, engine, test_results, export_path = run_training(config, logger=mlflow_logger)
        
        # F. Run predictions on test set to extract raw outputs for visualization
        print("\n--- Generating Custom Visualization Dashboard Elements ---")
        predictions = engine.predict(model=model, datamodule=datamodule)
        
        all_gt = []
        all_scores = []
        
        visualization_count = 0
        max_visualizations = mlflow_config.get("max_visualizations", 15)
        
        for batch in predictions:
            # Helper to parse batch outputs safely
            def to_list_or_array(item):
                if item is None:
                    return []
                if hasattr(item, "cpu"):
                    item = item.cpu()
                if hasattr(item, "numpy"):
                    item = item.numpy()
                if isinstance(item, np.ndarray):
                    return item.tolist() if item.ndim == 1 else item
                if isinstance(item, (list, tuple)):
                    return list(item)
                return [item]

            def get_batch_field(batch_obj, name):
                if batch_obj is None:
                    return None
                if isinstance(batch_obj, dict):
                    return batch_obj.get(name)
                if hasattr(batch_obj, name):
                    return getattr(batch_obj, name)
                if isinstance(batch_obj, (list, tuple)):
                    try:
                        return batch_obj[name]
                    except Exception:
                        return None
                return None
                
            image_paths = to_list_or_array(get_batch_field(batch, "image_path"))
            pred_scores = to_list_or_array(get_batch_field(batch, "pred_score"))
            pred_labels = to_list_or_array(get_batch_field(batch, "pred_label"))
            gt_labels = to_list_or_array(get_batch_field(batch, "gt_label"))
            
            # Fallback for older versions if gt_label is missing
            if not gt_labels and get_batch_field(batch, "label") is not None:
                gt_labels = to_list_or_array(get_batch_field(batch, "label"))
            
            raw_images = get_batch_field(batch, "image")
            raw_heatmaps = get_batch_field(batch, "anomaly_map")
            raw_masks = get_batch_field(batch, "pred_mask")
            
            if hasattr(raw_images, "cpu"):
                raw_images = raw_images.cpu().numpy()
            if hasattr(raw_heatmaps, "cpu"):
                raw_heatmaps = raw_heatmaps.cpu().numpy()
            if hasattr(raw_masks, "cpu"):
                raw_masks = raw_masks.cpu().numpy()
                
            B = len(image_paths)
            for idx in range(B):
                gt_val = int(gt_labels[idx]) if idx < len(gt_labels) else 0
                score_val = float(pred_scores[idx]) if idx < len(pred_scores) else 0.0
                pred_val = int(pred_labels[idx]) if idx < len(pred_labels) else 0
                
                all_gt.append(gt_val)
                all_scores.append(score_val)
                
                # Check if we should log this image collage
                if visualization_count < max_visualizations and mlflow_config.get("log_images", True):
                    img_slice = raw_images[idx]
                    
                    # Permute dims if channels first [C, H, W] -> [H, W, C]
                    if img_slice.ndim == 3 and img_slice.shape[0] in (1, 3):
                        img_slice = np.transpose(img_slice, (1, 2, 0))
                        
                    # Normalize if standard 0-1 float
                    if img_slice.dtype in (np.float32, np.float64) and img_slice.max() <= 1.0:
                        img_slice = (img_slice * 255.0).astype(np.uint8)
                        
                    heatmap_slice = raw_heatmaps[idx]
                    if heatmap_slice.ndim == 3:
                        heatmap_slice = heatmap_slice.squeeze(0)
                        
                    mask_slice = raw_masks[idx]
                    if mask_slice.ndim == 3:
                        mask_slice = mask_slice.squeeze(0)
                        
                    # Generate blend overlay
                    gray_img = cv2.cvtColor(img_slice, cv2.COLOR_RGB2GRAY)
                    gray_img_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
                    
                    heatmap_norm = (heatmap_slice - heatmap_slice.min()) / (heatmap_slice.max() - heatmap_slice.min() + 1e-8)
                    heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                    
                    # overlay_slice = cv2.addWeighted(gray_img_rgb, 0.65, heatmap_color_rgb, 0.35, 0)
                    # 1. Ensure both images are the exact same size (just in case)
                    if gray_img_rgb.shape != heatmap_color_rgb.shape:
                        heatmap_color_rgb = cv2.resize(heatmap_color_rgb, (gray_img_rgb.shape[1], gray_img_rgb.shape[0]))

                    # 2. Ensure both images are uint8 (0-255) data type
                    # (If your heatmap is 0.0 to 1.0, multiply by 255 first: (heatmap * 255).astype(np.uint8))
                    gray_img_rgb = gray_img_rgb.astype(np.uint8)
                    heatmap_color_rgb = heatmap_color_rgb.astype(np.uint8)

                    # 3. Now the blend will work perfectly
                    overlay_slice = cv2.addWeighted(gray_img_rgb, 0.65, heatmap_color_rgb, 0.35, 0)
                    
                    threshold = getattr(model, "image_threshold", getattr(model, "threshold", 0.5))
                    if hasattr(threshold, "cpu"):
                        threshold = threshold.cpu().item()
                    threshold = float(threshold)
                    
                    from src.utils import create_overlay_collage
                    collage_path = f"./results/temp_collage_{visualization_count}.png"
                    
                    create_overlay_collage(
                        image=img_slice,
                        heatmap=heatmap_norm,
                        mask=mask_slice,
                        overlay=overlay_slice,
                        gt_label=gt_val,
                        pred_label=pred_val,
                        score=score_val,
                        threshold=threshold,
                        output_path=collage_path
                    )
                    
                    # Upload to MLflow
                    mlflow.log_image(
                        Image.open(collage_path),
                        f"predictions/collage_{visualization_count:03d}_label_{gt_val}.png"
                    )
                    
                    # Remove temp file
                    if os.path.exists(collage_path):
                        os.remove(collage_path)
                        
                    visualization_count += 1
                    
        # G. Calculate advanced classification performance curves and metrics
        from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, average_precision_score
        y_true = np.array(all_gt)
        y_scores = np.array(all_scores)
        
        # Calculate thresholds
        threshold = getattr(model, "image_threshold", getattr(model, "threshold", 0.5))
        if hasattr(threshold, "cpu"):
            threshold = threshold.cpu().item()
        threshold = float(threshold)
        
        y_pred = (y_scores >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Calculate AUC/AP
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        # Compute secondary metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tn + fp + fn + tp + 1e-8)
        
        print("\n--- Performance Metrics Summary ---")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")
        print(f"  PR AP:     {ap:.4f}")
        
        # Log all of these directly as metrics under the active run
        mlflow.log_metrics({
            "eval_tn": int(tn),
            "eval_fp": int(fp),
            "eval_fn": int(fn),
            "eval_tp": int(tp),
            "eval_accuracy": float(accuracy),
            "eval_precision": float(precision),
            "eval_recall": float(recall),
            "eval_f1": float(f1),
            "eval_roc_auc": float(roc_auc),
            "eval_pr_ap": float(ap),
            "eval_threshold": float(threshold)
        })
        
        # H. Plot & Log Confusion Matrix Figure
        from src.utils import plot_confusion_matrix, plot_roc_pr_curves
        
        cm_path = "./results/confusion_matrix.png"
        plot_confusion_matrix(tn, fp, fn, tp, threshold=threshold, output_path=cm_path)
        mlflow.log_artifact(cm_path, "evaluation_charts")
        if os.path.exists(cm_path):
            os.remove(cm_path)
            
        # I. Plot & Log ROC/PR performance curves
        curves_path = "./results/performance_curves.png"
        plot_roc_pr_curves(fpr, tpr, roc_auc, precisions, recalls, ap, output_path=curves_path)
        mlflow.log_artifact(curves_path, "evaluation_charts")
        if os.path.exists(curves_path):
            os.remove(curves_path)
            
        # J. Log Exported openvino directories to MLflow
        if export_path and os.path.exists(export_path):
            print(f"\nLogging exported models from {export_path} to MLflow...")
            # export_path is the XML file, we want to log the folder containing it
            model_dir = pathlib.Path(export_path).parent
            mlflow.log_artifacts(str(model_dir), "production_models")
            
        print("\n==============================================================================")
        print("[SUCCESS] End-to-end Advanced Anomalib Training Pipeline Complete!")
        print("All metrics, custom diagrams, prediction collages, and files logged to MLflow.")
        print("==============================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Anomalib end-to-end training entrypoint.")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/default.yaml",
        help="Path to YAML configuration settings."
    )
    args = parser.parse_args()
    main(args)
