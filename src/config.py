import os


_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(_BASE_DIR, "mlp.npy")
DATA_CSV_PATH = os.path.join(_BASE_DIR, "data.csv")
TRAIN_CSV_PATH = os.path.join(_BASE_DIR, "training.csv")
PREDICT_CSV_PATH = os.path.join(_BASE_DIR, "predict.csv")

LEARNING_RATE = 0.03
EPOCHS = 1500
LAYERS = [30, 16, 8, 2]
PATIENCE = 20
BATCH_SIZE = 32

# L'OPTIMAZER Nesterov Momentum
MOMENTUM = 0.7
USE_NESTEROV = True

DATA_SPLIT = 0.7
