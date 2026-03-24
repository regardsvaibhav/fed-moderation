"""
Central configuration for the Federated Content Moderation System.
All settings loaded from .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Federated Learning ─────────────────────────────
FL_SERVER_ADDRESS = os.getenv("FL_SERVER_ADDRESS", "localhost:8080")
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", 3))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", 3))
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# Opacus Differential Privacy
DP_MAX_GRAD_NORM = 1.0
DP_NOISE_MULTIPLIER = 1.1
DP_DELTA = 1e-5

# ── Model ───────────────────────────────────────────
MAX_SEQ_LEN = 128
VOCAB_SIZE = 10000
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_CLASSES = 2           # 0 = safe, 1 = toxic

# ── Data ────────────────────────────────────────────
DATA_RAW_PATH = "data/raw"
DATA_PROCESSED_PATH = "data/processed"
DATASET_NAME = "hate_speech_offensive"  # HuggingFace dataset

# ── GenAI ───────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"

# ── IPFS / Pinata ───────────────────────────────────
PINATA_API_KEY = os.getenv("PINATA_API_KEY", "")
PINATA_SECRET_KEY = os.getenv("PINATA_SECRET_KEY", "")
PINATA_ENDPOINT = "https://api.pinata.cloud/pinning/pinJSONToIPFS"

# ── MLflow ──────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlflow_tracking")
MLFLOW_EXPERIMENT = "federated-content-moderation"

# ── API ─────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Labels ──────────────────────────────────────────
LABEL_MAP = {0: "SAFE", 1: "TOXIC"}
LABEL_COLORS = {"SAFE": "green", "TOXIC": "red"}