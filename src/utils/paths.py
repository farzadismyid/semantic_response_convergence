from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
LOG_DIR = DATA_DIR / "logs"
KB_DIR = DATA_DIR / "kb"

CONFIG_DIR = PROJECT_ROOT / "src" / "configs"