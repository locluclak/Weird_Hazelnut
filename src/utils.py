# anomalib_advanced/src/utils.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless mode to prevent GUI thread issues
import matplotlib.pyplot as plt
from PIL import Image

def plot_confusion_matrix(tn: int, fp: int, fn: int, tp: int, threshold: float = None, output_path: str = None):
    """
    Generate a high-end, premium style Confusion Matrix.
    Uses custom cool-toned palette (Deep Indigo, Muted Teal, Lavender) for modern aesthetics.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    # Custom 2x2 confusion matrix array
    cm = np.array([[tn, fp],
                   [fn, tp]])
    
    # Colors
    cmap = plt.cm.Purples  # A modern violet/purple theme
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    
    # Add beautiful custom colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.yaxis.get_offset_text().set_fontsize(9)
    
    # Labels and ticks
    classes = ['Good (Normal)', 'Anomaly']
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=10, fontweight='bold', color='#2c3e50')
    ax.set_yticklabels(classes, fontsize=10, fontweight='bold', color='#2c3e50', rotation=90, va="center")
    
    # Grid and spines styling
    for edge, spine in ax.spines.items():
        spine.set_color('#bdc3c7')
        spine.set_linewidth(1.2)
        
    # Set text inside cells
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "#2c3e50"
            text_str = f"{cm[i, j]}"
            
            # Sub-label for context (TN, FP, FN, TP)
            if i == 0 and j == 0:
                label = "\n(True Neg)"
            elif i == 0 and j == 1:
                label = "\n(False Pos)"
            elif i == 1 and j == 0:
                label = "\n(False Neg)"
            else:
                label = "\n(True Pos)"
                
            ax.text(j, i, f"{text_str}{label}",
                    ha="center", va="center",
                    color=color, fontsize=12, fontweight='bold')
            
    # Set titles and labels
    ax.set_ylabel('Ground Truth', fontsize=11, fontweight='bold', labelpad=10, color='#34495e')
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold', labelpad=10, color='#34495e')
    
    title = 'Evaluation Confusion Matrix'
    if threshold is not None:
        title += f"\n(Threshold = {threshold:.4f})"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return output_path
    else:
        return fig

def plot_roc_pr_curves(fpr, tpr, roc_auc, precisions, recalls, average_precision, output_path: str = None):
    """
    Plots professional ROC and PR curves side by side with modern styling.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # 1. ROC Curve
    ax1.plot(fpr, tpr, color='#8e44ad', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='#bdc3c7', lw=1.5, linestyle='--')
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel('False Positive Rate', fontsize=10, fontweight='bold', color='#34495e', labelpad=8)
    ax1.set_ylabel('True Positive Rate', fontsize=10, fontweight='bold', color='#34495e', labelpad=8)
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=11, fontweight='bold', pad=12, color='#2c3e50')
    ax1.legend(loc="lower right", fontsize=9, frameon=True, edgecolor='#e2e8f0')
    ax1.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
    
    # Beautify spines
    for spine in ax1.spines.values():
        spine.set_color('#cbd5e1')
        
    # 2. Precision-Recall Curve
    ax2.plot(recalls, precisions, color='#16a085', lw=2.5, label=f'PR Curve (AP = {average_precision:.4f})')
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_xlabel('Recall', fontsize=10, fontweight='bold', color='#34495e', labelpad=8)
    ax2.set_ylabel('Precision', fontsize=10, fontweight='bold', color='#34495e', labelpad=8)
    ax2.set_title('Precision-Recall Curve (PR)', fontsize=11, fontweight='bold', pad=12, color='#2c3e50')
    ax2.legend(loc="lower left", fontsize=9, frameon=True, edgecolor='#e2e8f0')
    ax2.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
    
    # Beautify spines
    for spine in ax2.spines.values():
        spine.set_color('#cbd5e1')
        
    plt.suptitle('Performance Curves Summary', fontsize=14, fontweight='bold', color='#2c3e50', y=1.02)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return output_path
    else:
        return fig

def create_overlay_collage(
    image: np.ndarray,
    heatmap: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
    gt_label: int,
    pred_label: int,
    score: float,
    threshold: float,
    output_path: str = None
):
    """
    Creates a beautiful 1x4 horizontal collage of the predictions:
    [Original Image] | [Anomaly Heatmap] | [Predicted Mask] | [Overlay]
    And decorates it with high-end, premium telemetry labels at the top.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=300)
    
    # Clear backgrounds and ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
            
    # Subplot 1: Original Image
    # Normalise if float
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() > 1.0:
            image = image / 255.0
    axes[0].imshow(image)
    axes[0].set_title('Input Image', fontsize=11, fontweight='bold', pad=8, color='#2c3e50')
    
    # Subplot 2: Anomaly Heatmap
    # Normalize heatmap if needed
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    axes[1].imshow(heatmap_norm, cmap='plasma')
    axes[1].set_title('Anomaly Heatmap', fontsize=11, fontweight='bold', pad=8, color='#2c3e50')
    
    # Subplot 3: Segmentation Mask
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Segmentation Mask', fontsize=11, fontweight='bold', pad=8, color='#2c3e50')
    
    # Subplot 4: Overlay
    # If overlay is already generated, use it, else draw heatmap on grayscale image
    if overlay.dtype == np.float32 or overlay.dtype == np.float64:
        if overlay.max() > 1.0:
            overlay = overlay / 255.0
    axes[3].imshow(overlay)
    axes[3].set_title('Anomaly Overlay', fontsize=11, fontweight='bold', pad=8, color='#2c3e50')
    
    # Create stylish header/telemetry
    gt_str = "ANOMALY" if gt_label == 1 else "GOOD"
    pred_str = "ANOMALY" if pred_label == 1 else "GOOD"
    
    # High-contrast color flags for status
    gt_color = "#e74c3c" if gt_label == 1 else "#2ecc71"
    pred_color = "#e74c3c" if pred_label == 1 else "#2ecc71"
    
    # Text overlay in center of the figure top
    status_text = (
        f"GT Status: {gt_str}  |  "
        f"Pred Status: {pred_str}  |  "
        f"Anomaly Score: {score:.4f}  (Thresh: {threshold:.4f})"
    )
    
    # Add border and background around plot by adding a beautiful title
    match_status = "CORRECT" if gt_label == pred_label else "MISCLASSIFIED"
    match_color = "#27ae60" if gt_label == pred_label else "#d35400"
    
    fig.suptitle(
        f"{status_text}   >>   [{match_status}]",
        fontsize=13, fontweight='bold',
        color='#fff', backgroundcolor='#2c3e50',
        y=0.98, x=0.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='#2c3e50', edgecolor='#34495e', lw=1.5)
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return output_path
    else:
        return fig
