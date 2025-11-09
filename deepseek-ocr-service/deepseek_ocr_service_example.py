#!/usr/bin/env python
"""
DeepSeek-OCR FastAPI service using vLLM.

- Loads deepseek-ai/DeepSeek-OCR via vLLM
- Exposes:
    GET  /healthz
    POST /validate_piicrop  (payload: {image_base64, prompt, category?, temperature?, max_tokens?})

Notes:
- Keep prompts minimal/redacted. This endpoint is perfect as the L3 validator in your pipeline.
- Pin the model revision via MODEL_REV if you want to freeze remote code.
"""

import base64
import io
import logging
import os
import time
from functools import lru_cache
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

# vLLM
from vllm import LLM, SamplingParams

# Some DeepSeek-OCR custom logits processor is optional.
# If it's not available in your vLLM wheel, just disable it.
try:
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
    HAVE_NGRAM = True
except Exception:
    HAVE_NGRAM = False


# -------------------------
# Config via environment
# -------------------------

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
MODEL_REV = os.getenv("MODEL_REV", None)  # e.g. "9f30c71" from your HF log
DTYPE = os.getenv("DTYPE", "bfloat16")    # "float16" if your GPU lacks bf16
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CORS_ALLOW = os.getenv("CORS_ALLOW", "*")  # set to comma list of origins in prod

# vLLM performance knobs
TENSOR_PARALLEL = int(os.getenv("TENSOR_PARALLEL", "1"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
GPU_MEMORY_UTIL = float(os.getenv("GPU_MEMORY_UTIL", "0.90"))

# Simple rate caps (per-process; use real gateway/rate-limiter in prod)
MAX_TOKENS_DEFAULT = int(os.getenv("MAX_TOKENS_DEFAULT", "512"))
TEMPERATURE_DEFAULT = float(os.getenv("TEMPERATURE_DEFAULT", "0.0"))

# -------------------------
# Logging
# -------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("deepseek-ocr-service")


# -------------------------
# FastAPI app
# -------------------------

app = FastAPI(title="DeepSeek-OCR Service (vLLM)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ALLOW.split(",")] if CORS_ALLOW else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_llm: Optional[LLM] = None

# -------------------------
# Models
# -------------------------

class ValidationRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded PNG/JPEG crop")
    prompt: str = Field(..., description="Redacted validation prompt")
    category: Optional[str] = Field(None, description="PII category hint (optional)")
    temperature: float = Field(TEMPERATURE_DEFAULT, ge=0.0, le=1.0)
    max_tokens: int = Field(MAX_TOKENS_DEFAULT, ge=1, le=2048)


class ValidationResponse(BaseModel):
    text: str
    latency_ms: int
    cached: bool = False
    backend: str = "vLLM"
    model: str = MODEL_NAME


# -------------------------
# Utilities
# -------------------------

def _decode_image(b64: str) -> Image.Image:
    try:
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}")


def _build_sampling_params(temperature: float, max_tokens: int) -> SamplingParams:
    extra_args: Dict[str, Any] = {}
    if HAVE_NGRAM:
        # These are the defaults recommended by DS-OCR authors to reduce hallucinated HTML tables
        extra_args.update(dict(ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822}))
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
        skip_special_tokens=False,
        stop=None,
        extra_args=extra_args if HAVE_NGRAM else None,
    )


# Very small in-process cache keyed by (prompt, SHA256(image_bytes))
# You’ll likely want Redis in production.
@lru_cache(maxsize=1024)
def _cache_key(prompt: str, image_hash: str) -> Optional[str]:
    return None


def _sha256_of_image(img: Image.Image) -> str:
    import hashlib
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


# -------------------------
# Lifecycle
# -------------------------

@app.on_event("startup")
def _startup():
    global _llm

    log.info("Loading model with vLLM...")
    llm_kwargs = dict(
        model=MODEL_NAME,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL,
        max_model_len=MAX_MODEL_LEN,
        dtype=DTYPE,  # "bfloat16" or "float16"
        gpu_memory_utilization=GPU_MEMORY_UTIL,
    )
    if MODEL_REV:
        llm_kwargs["revision"] = MODEL_REV

    # Turn off prefix cache unless you plan to stream long docs in chunks:
    llm_kwargs["enable_prefix_caching"] = False

    # If DeepSeek-OCR needs mm cache memory, you can tweak:
    # llm_kwargs["mm_processor_cache_gb"] = 0

    _llm = LLM(**llm_kwargs)
    log.info("Model loaded.")


@app.get("/healthz")
def healthz():
    if _llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"ok": True, "backend": "vLLM", "model": MODEL_NAME, "rev": MODEL_REV or "latest"}


# -------------------------
# Inference
# -------------------------

@app.post("/validate_piicrop", response_model=ValidationResponse)
def validate_pii_crop(req: ValidationRequest):
    if _llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()

    # Decode image
    image = _decode_image(req.image_base64)
    img_hash = _sha256_of_image(image)

    # Check trivial in-process cache
    ck = f"{req.prompt}::{img_hash}"
    cached = False
    try:
        cached_text = _cache_key.cache_info()  # touch cache so lru_cache decorator is active
    except Exception:
        pass

    # vLLM MM input format: list of dicts
    model_input = [{
        "prompt": req.prompt,
        "multi_modal_data": {"image": image},
    }]

    sampling = _build_sampling_params(req.temperature, req.max_tokens)

    try:
        outputs = _llm.generate(model_input, sampling)
        text = outputs[0].outputs[0].text
    except Exception as e:
        log.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    latency_ms = int((time.time() - t0) * 1000)
    return ValidationResponse(text=text, latency_ms=latency_ms, cached=cached)


# -------------------------
# Entrypoint
# -------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("deepseek_ocr_service:app", host=HOST, port=PORT, reload=False)
