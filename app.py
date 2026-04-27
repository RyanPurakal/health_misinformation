from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "claim_model.joblib"

app = FastAPI(title="Health Claim Checker")


class ClaimRequest(BaseModel):
    claim: str


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run `python3 train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


model = None


@app.on_event("startup")
def startup_event():
    global model
    model = load_model()


@app.post("/api/predict")
def predict(payload: ClaimRequest):
    text = payload.claim.strip()
    if not text:
        return {"error": "Claim text is required."}

    probs = model.predict_proba([text])[0]
    pred = int(model.predict([text])[0])  # 1 misinformation, 0 reliable
    misinformation_prob = float(probs[1])
    label = "MISINFORMATION" if pred == 1 else "RELIABLE"
    confidence = misinformation_prob if pred == 1 else 1.0 - misinformation_prob

    if pred == 1:
        explanation = "The model found language patterns commonly seen in misleading claims."
    else:
        explanation = "The model found language patterns more consistent with reliable claims."

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "misinformation_probability": round(misinformation_prob, 3),
        "explanation": explanation,
    }


@app.get("/")
def root():
    return FileResponse(BASE_DIR / "frontend" / "index.html")


app.mount("/frontend", StaticFiles(directory=BASE_DIR / "frontend", html=True), name="frontend")
