from pathlib import Path

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_INITIAL = 20
EPOCHS_FINE_TUNE = 10

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "Brain-Tumor" / "Training"
TEST_DIR = BASE_DIR / "Brain-Tumor" / "Testing"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "mobilenet_model.h5"

OUTPUT_DIR = BASE_DIR / "outputs"
GRADCAM_DIR = OUTPUT_DIR / "gradcam_results"
