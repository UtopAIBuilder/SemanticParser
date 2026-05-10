"""
api/app.py

FastAPI endpoint for C function analysis.

POST /analyze
  Body: {"code": "void init_driver() { ... }"}
  Returns: structured JSON with labels, embeddings, predictions

Run:
    uvicorn api.app:app --reload --port 8000
"""

import json
import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# Add src/ to path so we can import from sibling modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from label import (
    extract_call_names_from_code,
    deterministic_side_effects,
    llm_side_effects_fallback,
    llm_high_level_purpose,
)
from infer import (
    extract_functions_from_file,
    embed_code,
    fuse_embedding,
    classify,
    _load_model,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_NAME = "microsoft/unixcoder-base"
MAX_LENGTH = 512
AST_FEATURE_ORDER = [
    "if_count", "loop_count", "return_count", "call_count",
    "ast_depth", "line_count", "param_count", "pointer_count",
]

app = FastAPI(
    title="H2Loop Function Analyzer",
    description="Analyze embedded C functions: side effects, purpose, embeddings.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: load artifacts once
# ---------------------------------------------------------------------------

_scaler = None
_clf_payload = None


@app.on_event("startup")
async def startup_event():
    global _scaler, _clf_payload

    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    clf_path = ARTIFACTS_DIR / "classifier.pkl"

    if not scaler_path.exists():
        log.warning("scaler.pkl not found at %s. /analyze will fail until eval.py is run.", scaler_path)
    else:
        with open(scaler_path, "rb") as f:
            _scaler = pickle.load(f)
        log.info("Loaded scaler from %s", scaler_path)

    if not clf_path.exists():
        log.warning("classifier.pkl not found at %s. /analyze will fail until eval.py is run.", clf_path)
    else:
        with open(clf_path, "rb") as f:
            _clf_payload = pickle.load(f)
        log.info("Loaded classifier from %s", clf_path)

    # Pre-load UniXcoder model
    _load_model()
    log.info("UniXcoder model loaded and ready")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CodeRequest(BaseModel):
    code: str


class FunctionResult(BaseModel):
    name: str
    high_level_purpose: str
    predicted_side_effects: list[str]
    control_flow_elements: list[str]
    ast_features: dict
    embedding_dim: int
    embedding_preview: list[float]


class AnalyzeResponse(BaseModel):
    functions: list[FunctionResult]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: CodeRequest):
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="code field must not be empty")

    if _scaler is None or _clf_payload is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Run fuse.py and eval.py first.",
        )

    # Write code to temp file for tree-sitter extraction
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False, encoding="utf-8") as tmp:
        tmp.write(request.code)
        tmp_path = Path(tmp.name)

    try:
        functions = extract_functions_from_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not functions:
        return AnalyzeResponse(functions=[])

    results = []
    for fn in functions:
        code = fn["function_code"]

        # Label
        call_names = extract_call_names_from_code(code)
        side_effects, is_ambiguous = deterministic_side_effects(call_names)
        if is_ambiguous:
            side_effects = llm_side_effects_fallback(code)
        try:
            high_level_purpose = llm_high_level_purpose(code)
        except Exception as exc:
            log.warning("LLM high_level_purpose failed for %s: %s", fn["function_name"], exc)
            high_level_purpose = "Unable to generate description"

        # Embed
        embedding = embed_code(code)

        # Fuse
        fused = fuse_embedding(embedding, fn["ast_features"], _scaler)

        # Classify
        predicted = classify(fused, _clf_payload)

        results.append(FunctionResult(
            name=fn["function_name"],
            high_level_purpose=high_level_purpose,
            predicted_side_effects=predicted,
            control_flow_elements=fn["control_flow_elements"],
            ast_features=fn["ast_features"],
            embedding_dim=len(fused),
            embedding_preview=embedding[:3],
        ))

    return AnalyzeResponse(functions=results)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "scaler_loaded": _scaler is not None,
        "classifier_loaded": _clf_payload is not None,
    }


@app.get("/")
async def root():
    return {"message": "H2Loop Function Analyzer API. POST /analyze with {'code': '<c_code>'}"}
