# anomalib_advanced/src/engine.py
from anomalib.engine import Engine
from anomalib.deploy import ExportType
from src.data import get_datamodule
from src.model import create_model

def run_training(config: dict, logger=None):
    """
    Run the end-to-end training and evaluation pipeline.
    
    Args:
        config (dict): Parsed configuration dictionary.
        logger (Any): Logger instance (e.g., AnomalibMLFlowLogger).
        
    Returns:
        tuple: (model, datamodule, engine, test_results, export_path)
    """
    # 1. Initialize Datamodule
    print("\n--- Initializing Dataset Datamodule ---")
    datamodule = get_datamodule(config)
    
    # 2. Initialize Model
    print("\n--- Initializing Anomaly Detection Model ---")
    model = create_model(config)
    
    # 3. Create Engine
    print("\n--- Initializing Engine ---")
    engine_args = {
        "max_epochs": config["engine"].get("max_epochs", 1),
        "devices": config["engine"].get("devices", 0),
        "accelerator": config["engine"].get("accelerator", "cpu")
    }
    
    # If logger is provided, inject it
    if logger is not None:
        engine_args["logger"] = logger
        
    engine = Engine(**engine_args)
    
    # 4. Train model (Fit)
    print("\n--- Training Model (Fit) ---")
    engine.fit(model=model, datamodule=datamodule)
    
    # 5. Evaluate model (Test)
    print("\n--- Evaluating Model on Test Set (Test) ---")
    test_results = engine.test(model=model, datamodule=datamodule)
    print(f"Test Evaluation Results: {test_results}")
    
    # 6. Export Model
    print("\n--- Exporting Model for Production ---")
    export_type_str = config["export"].get("export_type", "OPENVINO").upper()
    export_type = ExportType[export_type_str]
    export_root = config["export"].get("export_root", "./results/exported")
    
    export_path = engine.export(
        model=model,
        export_type=export_type,
        export_root=export_root,
        input_size=config["data"].get("image_size", (256, 256)),
        datamodule=datamodule
    )
    print(f"Model successfully exported to: {export_path}")
    
    return model, datamodule, engine, test_results, export_path
