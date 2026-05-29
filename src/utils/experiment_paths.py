from datetime import datetime
from pathlib import Path

from src.utils.paths import OUTPUT_DIR


def create_experiment_dir(experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = OUTPUT_DIR / f"{timestamp}_{experiment_name}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir
