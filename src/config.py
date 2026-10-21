from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
IMAGE_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
STATIC_DIR = PROJECT_ROOT / "static"

# --- Data ---
IMAGE_SIZE = 224
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 42

# --- Training ---
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 0  # 0 avoids multiprocessing problems on Windows