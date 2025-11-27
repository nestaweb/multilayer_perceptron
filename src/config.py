import os


_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATASET_DIR = os.path.join(_BASE_DIR, "dataset")

MODEL_PATH = os.path.join(_BASE_DIR, "mlp.npy")
DATA_CSV_PATH = os.path.join(_DATASET_DIR, "data.csv")
TRAIN_CSV_PATH = os.path.join(_DATASET_DIR, "training.csv")
PREDICT_CSV_PATH = os.path.join(_DATASET_DIR, "predict.csv")

