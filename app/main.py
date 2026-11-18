"""SafeNet API.  Run:  uvicorn app.main:app --reload"""
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import fusion
from .features import extract_urls, text_cues, url_features, url_red_flags

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

if not (MODELS / "text_model.joblib").exists() or not (MODELS / "url_model.joblib").exists():
    sys.path.insert(0, str(ROOT))
    import train
    train.main()

text_model = joblib.load(MODELS / "text_model.joblib")
url_model = joblib.load(MODELS / "url_model.joblib")

app = FastAPI(title="SafeNet", description="AI-powered phishing & scam detection")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class AnalyzeRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None


def score_url(u: str) -> float:
    return float(url_model.predict_proba(np.array([url_features(u)]))[0][1])


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    text = (req.text or "").strip()
    url = (req.url or "").strip()
    if not text and not url:
        raise HTTPException(400, "Provide text and/or a URL to analyze.")

    urls = list(dict.fromkeys(([url] if url else []) + extract_urls(text)))

    text_p = float(text_model.predict_proba([text])[0][1]) if text else None
    url_results = [{"url": u, "probability": round(score_url(u), 4), "red_flags": url_red_flags(u)} for u in urls]
    url_p = max((r["probability"] for r in url_results), default=None)

    cues = text_cues(text)
    score = fusion.fuse(text_p, url_p)
    # Heuristic safety net: several independent scam cues should never read as LOW.
    if len(cues) >= 3 and score < fusion.LOW_T + 0.05:
        score = fusion.LOW_T + 0.05
    level = fusion.risk_level(score)

    reasons = [f"{name}: {', '.join(sorted(set(matches))[:4])}" for name, matches in cues.items()]
    for r in url_results:
        reasons += [f"{r['url'][:60]}: {f}" for f in r["red_flags"] if r["probability"] >= 0.35]

    return {
        "risk_score": round(score * 100, 1),
        "risk_level": level,
        "threat_type": fusion.threat_type(level, text_p, url_p, cues),
        "recommendation": fusion.recommendation(level),
        "signals": {"text_probability": None if text_p is None else round(text_p, 4), "url_probability": url_p},
        "reasons": reasons,
        "urls_analyzed": url_results,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
