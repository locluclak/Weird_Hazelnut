# anomalib_advanced/src/model.py
from anomalib.models import Patchcore, Padim

def create_model(config: dict):
    """
    Dynamically instantiate the Anomalib model based on configuration.
    Currently supports:
      - Patchcore
      - Padim
    """
    model_config = config["model"]
    model_name = model_config["name"].lower()
    
    if model_name == "patchcore":
        # Extract Patchcore init args
        backbone = model_config.get("backbone", "resnet18")
        coreset_ratio = model_config.get("coreset_sampling_ratio", 0.01)
        neighbors = model_config.get("num_neighbors", 9)
        image_size = config["data"].get("image_size", (256, 256))
        
        print(f"Creating Patchcore model (backbone={backbone}, coreset_ratio={coreset_ratio}, neighbors={neighbors}, image_size={image_size})...")
        
        # Configure pre-processor with the specified image size
        pre_processor = Patchcore.configure_pre_processor(image_size=tuple(image_size))
        
        model = Patchcore(
            backbone=backbone,
            coreset_sampling_ratio=coreset_ratio,
            num_neighbors=neighbors,
            pre_processor=pre_processor
        )
        return model
        
    elif model_name == "padim":
        backbone = model_config.get("backbone", "resnet18")
        layers = model_config.get("layers", ["layer1", "layer2", "layer3"])
        image_size = config["data"].get("image_size", (256, 256))
        
        print(f"Creating Padim model (backbone={backbone}, layers={layers}, image_size={image_size})...")
        
        # Configure pre-processor with the specified image size
        pre_processor = Padim.configure_pre_processor(image_size=tuple(image_size))
        
        model = Padim(
            backbone=backbone,
            layers=layers,
            pre_processor=pre_processor
        )
        return model
        
    else:
        raise ValueError(
            f"Unsupported model name '{model_config['name']}'. "
            f"Please choose 'patchcore' or 'padim'."
        )
