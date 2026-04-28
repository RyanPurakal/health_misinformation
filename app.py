"""
FastAPI server for the web app layer. Loads the fine-tuned DistilBERT classifier
and TF-IDF explanation index once at startup (fail-fast if models/bert_model/ is
missing), then serves POST /api/predict and the static frontend. Response shape
is unchanged from the previous sklearn version so the frontend needs no edits.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models.bert_classifier import HealthClaimClassifier
from llm.explanation_generator import ExplanationGenerator


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Health Claim Checker")


class ClaimRequest(BaseModel):
    claim: str
    article_text: str = ""  # optional — accuracy is higher when provided


classifier: HealthClaimClassifier | None = None
explainer: ExplanationGenerator | None = None


@app.on_event("startup")
def startup_event():
    global classifier, explainer
    classifier = HealthClaimClassifier()
    explainer = ExplanationGenerator()


@app.post("/api/predict")
def predict(payload: ClaimRequest):
    claim = payload.claim.strip()
    if not claim:
        return {"error": "Claim text is required."}

    result = classifier.predict(claim, payload.article_text)
    explanation = explainer.get_explanation(claim, result["label_id"])
    misinformation_prob = result["probs"]["MISINFORMATION"]

    return {
        "label": result["label"],
        "confidence": round(result["confidence"], 3),
        "misinformation_probability": round(misinformation_prob, 3),
        "explanation": explanation,
    }


@app.get("/")
def root():
    return FileResponse(BASE_DIR / "frontend" / "index.html")


app.mount("/frontend", StaticFiles(directory=BASE_DIR / "frontend", html=True), name="frontend")
