# Anomalib Advanced: Anomaly Detection Pipeline

A production-ready anomaly detection pipeline using [Anomalib](https://github.com/openvinotoolkit/anomalib), [OpenVINO](https://docs.openvino.ai/), and [FastAPI](https://fastapi.tiangolo.com/). This project features end-to-end training, MLflow tracking, high-performance inference, and a web-service API.

## 🚀 Features
- **Training**: Advanced PatchCore/Padim models with automated data leakage detection.
- **Monitoring**: Deep integration with MLflow for tracking metrics, confusion matrices, and prediction collages.
- **Inference**: Optimized OpenVINO execution for low-latency production environments.
- **API**: Modern FastAPI service for real-time anomaly detection.

---

## 🛠️ Setup

### 1. Environment Installation
It is recommended to use a Conda environment with Python 3.12:

```bash
conda create -n adnut python=3.12 -y
conda activate adnut
pip install -r requirements.txt
```

### 2. Dataset Preparation
This repository does **not** include the dataset. You must download the MVTec Hazelnut dataset and organize it as follows:
```
data/
└── hazelnut/
    ├── train/
    │   └── good/ 391 images
    └── test/
        ├── good/ 40 images
        ├── crack/ 18 images
        ├── cut/ 17 images
        ├── hole/ 18 images
        └── print/ 17 images
```
Update the path in `config/default.yaml` under `data.root` to point to your `hazelnut` folder.

### 3. Data Versioning (DVC)
We use DVC to track dataset versions without bloating the Git repository.

```bash
# Initialize DVC (already done in this repo)
# dvc init

# Track your dataset
dvc add data/hazelnut

# Commit the .dvc file to Git
git add data/hazelnut.dvc .gitignore
git commit -m "Add dataset tracking via DVC"
```

---

## 📂 Project Structure
```text
anomalib_advanced/
├── config/             # YAML configuration files
├── src/                # Core logic (data, engine, model, inference)
├── train.py            # Training entry point
├── predict.py          # CLI Inference entry point
├── serve.py            # FastAPI service
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

---

## 🏋️ Training

Train the model and export it to OpenVINO format automatically:

```bash
python train.py --config ./config/default.yaml
```

- **MLflow Tracking**: Training metrics and visual results are logged to `http://127.0.0.1:5000`.
- **Export**: The OpenVINO model (`model.xml`, `model.bin`) will be saved in `results/exported/weights/openvino/`.

---

## 🔍 Inference (CLI)

Run batch inference on a single image or a folder of images using the exported OpenVINO model:

```bash
python predict.py --model results/exported/weights/openvino/model.xml --image ./crack/000.png --output ./results/predictions
```

**Key Arguments:**
- `--image`: Path to image or directory.
- `--device`: `CPU`, `GPU`, or `AUTO`.
- `--mlflow-log`: Enable this flag to log inference results and latency to a new MLflow run.

---

## 🌐 API Deployment

Expose the anomaly detection model as a high-performance web service.

### 1. Start the Server
```bash
# Set environment variables (optional)
# set MODEL_PATH=results/exported/weights/openvino/model.xml
# set INFERENCE_DEVICE=CPU
python serve.py
```
The API will be available at `http://localhost:8000`.

### 2. Test the API
You can test the `/predict` endpoint using `curl`:

```bash
curl.exe -X POST "http://localhost:8000/predict" -F "file=@./crack/000.png"
```

**Response Example:**
```json
{
  "filename": "000.png",
  "anomaly_score": 0.7342,
  "pred_label": 1,
  "label_name": "ANOMALY",
  "latency_ms": 191.5
}
```

---

## 📊 Dashboard & Monitoring

Launch the MLflow UI to compare runs, view performance curves (ROC/PR), and inspect prediction collages:

```bash
mlflow ui
```

Navigate to `http://127.0.0.1:5000` to see:
- **Metrics**: Accuracy, F1-Score, AUC, AP, and Latency.
- **Artifacts**: Confusion Matrix, ROC/PR curves, and visual overlays (Original + Heatmap + Mask).

---

## ⚙️ Configuration
Modify `config/default.yaml` to adjust:
- **Model**: Backbone selection (`resnet18`, `wide_resnet50_2`), coreset sampling ratio.
- **Hyperparameters**: Batch size, epochs, image size (default 256x256).
- **Logging**: MLflow experiment names and visualization limits.
