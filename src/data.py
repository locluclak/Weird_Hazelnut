# anomalib_advanced/src/data.py
import hashlib
import pathlib
from typing import Dict, Iterable, List, Tuple
from anomalib.data import Folder

def iter_images(root: pathlib.Path) -> Iterable[pathlib.Path]:
    """Iterate over all images in the root directory and subdirectories."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            yield path

def hash_file(path: pathlib.Path) -> str:
    """Calculate MD5 hash of a file to check for duplicates."""
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def detect_data_leakage(data_root_str: str) -> List[Tuple[pathlib.Path, pathlib.Path]]:
    """
    Check if any images in the train set are also in the test set.
    Returns:
        List of tuples: [(train_image_path, test_image_path)]
    """
    data_root = pathlib.Path(data_root_str).resolve()
    train_root = data_root / "train"
    test_root = data_root / "test"

    if not train_root.exists() or not test_root.exists():
        return []

    # Map train hashes to image paths
    train_hashes: Dict[str, pathlib.Path] = {}
    for img_path in iter_images(train_root):
        train_hashes[hash_file(img_path)] = img_path

    # Check test hashes against train hashes
    duplicates: List[Tuple[pathlib.Path, pathlib.Path]] = []
    for img_path in iter_images(test_root):
        digest = hash_file(img_path)
        if digest in train_hashes:
            duplicates.append((train_hashes[digest], img_path))

    return duplicates

def get_dataset_stats(data_root_str: str) -> Dict[str, int]:
    """Calculate image counts per directory for training and testing."""
    data_root = pathlib.Path(data_root_str).resolve()
    stats = {}
    
    # Train stats
    train_root = data_root / "train"
    if train_root.exists():
        for category in train_root.iterdir():
            if category.is_dir():
                count = len(list(iter_images(category)))
                stats[f"train/{category.name}"] = count

    # Test stats
    test_root = data_root / "test"
    if test_root.exists():
        for category in test_root.iterdir():
            if category.is_dir():
                count = len(list(iter_images(category)))
                stats[f"test/{category.name}"] = count
                
    return stats

def get_datamodule(config: dict) -> Folder:
    """
    Build the Anomalib Folder Datamodule using config parameters.
    """
    data_config = config["data"]
    
    # Map configurations safely to the Folder datamodule signature
    datamodule = Folder(
        name=data_config["name"],
        root=data_config["root"],
        normal_dir=data_config["normal_dir"],
        normal_test_dir=data_config["normal_test_dir"],
        abnormal_dir=data_config["abnormal_dir"],
        train_batch_size=data_config.get("train_batch_size", 32),
        eval_batch_size=data_config.get("test_batch_size", 32),  # mapping test_batch_size to eval_batch_size
        num_workers=data_config.get("num_workers", 0)
    )
    
    return datamodule
