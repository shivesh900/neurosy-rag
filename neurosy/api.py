"""FastAPI service.  Run: uvicorn neurosy.api:app --reload"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import DISCLAIMER, __version__
from .pipeline import get_engine

app = FastAPI(title="NeuroSy-RAG", version=__version__, description=DISCLAIMER)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class DiagnoseRequest(BaseModel):
    text: str = Field(..., min_length=3, examples=["high fever for 3 days with joint pain and a rash"])
    place: Optional[str] = Field(None, examples=["Tambaram, Chennai"])
    lat: Optional[float] = None
    lon: Optional[float] = None
    find_providers: bool = True


@app.get("/health")
def health():
    eng = get_engine()
    return {"status": "ok", "version": __version__, "model": eng.classifier.meta,
            "retriever": eng.explainer.retriever.backend, "diseases": len(eng.kb.diseases)}


@app.get("/diseases")
def diseases():
    kb = get_engine().kb
    return [{"key": k, "name": d.name, "icd10": d.icd10, "specialty": d.specialty, "risk_level": d.risk_level}
            for k, d in sorted(kb.diseases.items())]


@app.post("/diagnose")
def diagnose(req: DiagnoseRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is required")
    return get_engine().diagnose(req.text, place=req.place, lat=req.lat, lon=req.lon,
                                 find_providers=req.find_providers)
