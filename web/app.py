"""FastAPI server: API + static frontend for the health claim checker."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from healthchecker.service import ClaimCheckerService

STATIC_DIR = Path(__file__).resolve().parent / "static"
_service: ClaimCheckerService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    _service = ClaimCheckerService.load(verbose=True)
    yield
    _service = None


app = FastAPI(title="Health claim checker", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CheckBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)


@app.post("/api/check")
async def api_check(body: CheckBody):
    if _service is None:
        raise HTTPException(503, "Model not loaded yet.")
    return _service.analyze(body.text)


@app.get("/api/health")
async def health():
    return {"ok": _service is not None}


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
