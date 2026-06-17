"""
Standalone Chronos prediction microservice.
Lightweight - only depends on torch + chronos-forecasting + fastapi.
"""
import os
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chronos Prediction Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("CHRONOS_API_KEY", "").strip()

def verify_api_key(x_api_key: str = Header(None)):
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

_PIPELINE_CACHE = None

def get_pipeline():
    global _PIPELINE_CACHE
    if _PIPELINE_CACHE is None:
        import torch
        from chronos import BaseChronosPipeline
        logger.info("[Chronos] Loading model (first call only)...")
        _PIPELINE_CACHE = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        logger.info("[Chronos] Model loaded and cached.")
    return _PIPELINE_CACHE


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _PIPELINE_CACHE is not None}


@app.get("/chronos/prob")
def chronos_prob(closes: str, x_api_key: str = Header(None)):
    """
    Takes a comma-separated list of closing prices (oldest to newest),
    returns a probability of price going up over the next 5 steps.
    """
    verify_api_key(x_api_key)
    try:
        import torch

        prices = [float(p) for p in closes.split(",") if p.strip()]
        if len(prices) < 20:
            return {"prob": 0.5, "error": "Insufficient data (need at least 20 points)"}

        lookback = min(60, len(prices))
        closes_arr = np.array(prices[-lookback:])
        context = torch.tensor(closes_arr, dtype=torch.float32)

        pipeline = get_pipeline()
        forecast = pipeline.predict(context, prediction_length=5)
        samples = forecast[0].numpy()

        current_price = float(closes_arr[-1])
        up_samples = (samples[:, -1] > current_price).sum()
        total_samples = samples.shape[0]

        prob = float(up_samples / total_samples)
        dampened = 0.5 + (prob - 0.5) * 0.8

        median_forecast = np.median(samples, axis=0)
        predicted_price = float(median_forecast[-1])
        pct_change = (predicted_price - current_price) / current_price * 100

        return {
            "prob": round(dampened, 4),
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "predicted_pct_change": round(pct_change, 3),
            "up_probability": round(prob, 3),
            "down_probability": round(1 - prob, 3),
            "model": "chronos-t5-small",
        }
    except Exception as e:
        logger.error(f"Chronos error: {e}")
        return {"prob": 0.5, "error": str(e)}
