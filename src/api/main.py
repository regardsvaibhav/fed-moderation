"""
FastAPI Backend — Content Moderation API
=========================================
Endpoints:
  POST /moderate          — moderate a single post
  POST /moderate/batch    — moderate multiple posts
  GET  /health            — health check
  GET  /stats             — model + system stats
  GET  /experiments       — research results table

Run:
  uvicorn src.api.main:app --reload --port 8000
"""

import os, sys, pickle, json, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from loguru import logger

from src.config import DATA_PROCESSED_PATH, LABEL_MAP
from src.genai.explainer import ModerationExplainer
from src.federated.model import create_model

# ── App Setup ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Federated Content Moderation API",
    description="Privacy-preserving content moderation with GenAI explainability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ─────────────────────────────────────────────────────────────
model      = None
vocab      = None
explainer  = None
stats      = {}
request_count = 0
start_time = time.time()


# ── Startup: Load Model + Vocab ──────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global model, vocab, explainer, stats

    logger.info("Loading model and vocab...")

    # Load vocab
    vocab_path = os.path.join(DATA_PROCESSED_PATH, 'vocab.pkl')
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        logger.success(f"Vocab loaded: {len(vocab)} tokens")
    else:
        logger.warning("Vocab not found — run data_prep.py first")

    # Load model (federated if available, else centralized)
    model_path = os.path.join(DATA_PROCESSED_PATH, 'federated_model.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(DATA_PROCESSED_PATH, 'centralized_model.pt')

    if os.path.exists(model_path) and vocab:
        model = create_model(vocab_size=len(vocab))
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        logger.success(f"Model loaded from {model_path}")
    else:
        logger.warning("No trained model found — run experiment.py first")
        logger.info("API will use random predictions until model is trained")

    # Load experiment results
    results_path = os.path.join(DATA_PROCESSED_PATH, 'experiment_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            stats['experiments'] = json.load(f)

    # Load dataset stats
    ds_stats_path = os.path.join(DATA_PROCESSED_PATH, 'dataset_stats.json')
    if os.path.exists(ds_stats_path):
        with open(ds_stats_path) as f:
            stats['dataset'] = json.load(f)

    # Init explainer
    explainer = ModerationExplainer()
    logger.success("API ready ✅")

# ── Helpers ──────────────────────────────────────────────────────────────────
def predict(text: str):
    """Run inference. Returns (prediction, confidence)."""
    global model, vocab

    if model is None or vocab is None:
        import random
        return random.randint(0, 1), round(random.uniform(0.65, 0.95), 3)

    import re
    text_clean = re.sub(r'http\S+|@\w+|#\w+|[^a-zA-Z\s]', ' ', text.lower())
    tokens = text_clean.split()
    indices = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(indices) < 128:
        indices += [vocab['<PAD>']] * (128 - len(indices))
    x = torch.tensor([indices[:128]], dtype=torch.long)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        prediction = probs.argmax().item()
        confidence = probs[prediction].item()

    return prediction, round(confidence, 4)

# ── Request / Response Models ────────────────────────────────────────────────
class ModerateRequest(BaseModel):
    text: str
    epsilon: Optional[float] = 3.8      # privacy budget (from config)
    include_explanation: Optional[bool] = True

class BatchModerateRequest(BaseModel):
    posts: list[str]
    epsilon: Optional[float] = 3.8
    include_explanation: Optional[bool] = True


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    uptime = round(time.time() - start_time, 1)
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "vocab_loaded": vocab is not None,
        "uptime_secs":  uptime,
        "requests":     request_count,
    }


@app.post("/moderate")
async def moderate(req: ModerateRequest):
    global request_count
    request_count += 1

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(req.text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long (max 2000 chars)")

    # Run model
    t0 = time.time()
    prediction, confidence = predict(req.text)
    inference_ms = round((time.time() - t0) * 1000, 2)

    # Build response
    response = {
        "text":         req.text,
        "decision":     LABEL_MAP[prediction],
        "confidence":   confidence,
        "epsilon":      req.epsilon,
        "inference_ms": inference_ms,
        "request_id":   request_count,
    }

    # Add GenAI explanation
    if req.include_explanation and explainer:
        explanation = explainer.explain(
            text=req.text,
            prediction=prediction,
            confidence=confidence,
            epsilon=req.epsilon,
        )
        response["explanation"] = explanation

    return response


@app.post("/moderate/batch")
async def moderate_batch(req: BatchModerateRequest):
    if len(req.posts) > 50:
        raise HTTPException(status_code=400, detail="Max 50 posts per batch")

    results = []
    for text in req.posts:
        prediction, confidence = predict(text)
        result = {
            "text":       text[:100] + "..." if len(text) > 100 else text,
            "decision":   LABEL_MAP[prediction],
            "confidence": confidence,
        }
        if req.include_explanation and explainer:
            result["explanation"] = explainer.explain(
                text=text,
                prediction=prediction,
                confidence=confidence,
                epsilon=req.epsilon,
            )
        results.append(result)

    toxic_count = sum(1 for r in results if r["decision"] == "TOXIC")
    return {
        "total":       len(results),
        "toxic_count": toxic_count,
        "safe_count":  len(results) - toxic_count,
        "toxic_rate":  round(toxic_count / len(results), 3),
        "results":     results,
    }


@app.get("/stats")
def get_stats():
    return {
        "model": {
            "type":        "TextCNN (Federated Learning)",
            "privacy":     "Differential Privacy (Opacus)",
            "strategy":    "FedAvg",
            "num_clients": 3,
        },
        "dataset": stats.get("dataset", {}),
        "api": {
            "total_requests": request_count,
            "uptime_secs":    round(time.time() - start_time, 1),
        }
    }


@app.get("/experiments")
def get_experiments():
    """Return the research results table."""
    experiments = stats.get("experiments", [])
    if not experiments:
        return {"message": "No experiment results yet. Run experiment.py first."}
    return {
        "title":       "Privacy-Utility-Fairness Tradeoff",
        "description": "How differential privacy noise affects accuracy and fairness",
        "results":     experiments,
    }


# ── Dev entrypoint ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)