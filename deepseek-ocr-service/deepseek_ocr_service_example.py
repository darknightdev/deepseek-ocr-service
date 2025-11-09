"""
DeepSeek-OCR FastAPI Service (Transformers-only)

- Uses Hugging Face Transformers with trust_remote_code to load deepseek-ai/DeepSeek-OCR
- Works on CUDA GPUs (A10/T4/L4/A100/etc.). Falls back to CPU if needed.
- Picks dtype dynamically: bfloat16 if supported, else float16 on CUDA; float32 on CPU.
- Optional: pin a model revision via env var DEEPSEEK_OCR_REV to avoid remote-code drift.

Endpoints:
  GET  /healthz
  POST /validate_piicrop  -> { image_base64, prompt, category?, temperature?, max_tokens? }
"""

import base64
import io
import json
import logging
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

import torch
from transformers import AutoModel, AutoTokenizer, __version__ as hf_version


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL_ID = os.getenv("DEEPSEEK_OCR_MODEL", "deepseek-ai/DeepSeek-OCR")
MODEL_REV = os.getenv("DEEPSEEK_OCR_REV")  # e.g., "9f30c71..." commit hash to pin
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")  # comma-separated list if you want to restrict
CROP_BASE_SIZE = int(os.getenv("CROP_BASE_SIZE", "1024"))
CROP_IMAGE_SIZE = int(os.getenv("CROP_IMAGE_SIZE", "640"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))


# -----------------------------------------------------------------------------
# FastAPI app & CORS (tighten in prod)
# -----------------------------------------------------------------------------
app = FastAPI(title="DeepSeek-OCR PII Validation Service (Transformers)")

allow_origins = [o.strip() for o in ALLOW_ORIGINS.split(",")] if ALLOW_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("deepseek-ocr-service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# -----------------------------------------------------------------------------
# Request/Response models
# -----------------------------------------------------------------------------
class ValidationRequest(BaseModel):
    image_base64: str
    prompt: str
    category: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


class ValidationResponse(BaseModel):
    text: str
    output: Optional[str] = None
    latency_ms: int
    cached: bool = False
    backend: str = "transformers"


# -----------------------------------------------------------------------------
# Globals for model/tokenizer
# -----------------------------------------------------------------------------
_tokenizer = None
_model = None
_device = "cpu"
_dtype = torch.float32


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _select_device_and_dtype():
    """Choose device and dtype based on availability/capability."""
    if torch.cuda.is_available():
        device = "cuda"
        # T4 doesn't support bf16; A10/A100/L4 do. Pick the best we can.
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    return device, dtype


def _load_deepseek_ocr():
    """Load tokenizer & model via Transformers with trust_remote_code=True."""
    global _tokenizer, _model, _device, _dtype

    logger.info("Transformers version: %s", hf_version)
    _device, _dtype = _select_device_and_dtype()
    logger.info("Loading model on device=%s with dtype=%s", _device, str(_dtype).split(".")[-1])

    model_ref = MODEL_ID if not MODEL_REV else f"{MODEL_ID}@{MODEL_REV}"

    _tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        trust_remote_code=True
    )
    _model = AutoModel.from_pretrained(
        model_ref,
        trust_remote_code=True,
        torch_dtype=_dtype
    )

    # Move to device and eval
    if _device == "cuda":
        _model = _model.to(_dtype).cuda().eval()
    else:
        _model = _model.to(_dtype).eval()

    logger.info("DeepSeek-OCR loaded successfully: %s", model_ref)


def _decode_image(b64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")


# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------
@app.on_event("startup")
def _startup():
    try:
        logger.info("Loading model (Transformers-only)...")
        _load_deepseek_ocr()
    except Exception as e:
        logger.exception("Failed to load model")
        # Let startup fail loudly so orchestration can restart
        raise


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "backend": "transformers",
        "device": _device,
        "dtype": str(_dtype).split(".")[-1],
        "model": MODEL_ID,
        "revision": MODEL_REV or "latest",
    }


@app.post("/validate_piicrop", response_model=ValidationResponse)
def validate_piicrop(req: ValidationRequest):
    """
    Validate a PII candidate crop using DeepSeek-OCR.
    Expects a cropped image (PIL) & a compact instruction/prompt.

    Returns OCR/LLM text output; your upstream pipeline will parse/assess it.
    """
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    # Decode the image
    image = _decode_image(req.image_base64)

    # DeepSeek-OCR exposes .infer(tokenizer, prompt=..., image_file=..., ...)
    # We pass PIL Image directly; remote code supports it.
    try:
        res = _model.infer(
            _tokenizer,
            prompt=req.prompt,
            image_file=image,      # PIL Image supported by remote code
            output_path=None,
            base_size=CROP_BASE_SIZE,
            image_size=CROP_IMAGE_SIZE,
            crop_mode=True,
            save_results=False,
            test_compress=True
        )
        # The repo returns a dict containing "text" and/or "output"
        text = res.get("text") or res.get("output") or ""
    except Exception as e:
        logger.exception("DeepSeek-OCR inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    latency_ms = int((time.time() - start) * 1000)
    return ValidationResponse(
        text=text,
        output=text,
        latency_ms=latency_ms,
        cached=False,
        backend="transformers",
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
