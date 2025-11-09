"""
DeepSeek-OCR Transformers-only FastAPI Service (slow tokenizer; GPU-friendly)

Endpoints:
  GET  /healthz
  POST /validate_piicrop  { image_base64, prompt, category, temperature?, max_tokens? }

Notes:
- Forces slow (SentencePiece) tokenizer to avoid tokenizer.json fast-tokenizer errors.
- Picks dtype automatically: T4 => fp16; A10/A100/L4 => bfloat16; CPU => float32.
- Uses model's trust_remote_code .infer(...) method for OCR on image crops.
"""

import os
import io
import time
import json
import logging
import base64
from typing import Optional

# Force slow tokenizer + stable behavior
os.environ.setdefault("TRANSFORMERS_NO_FAST_TOKENIZER", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import AutoModel, AutoTokenizer, LlamaTokenizer, __version__ as HF_VER

# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger("deepseek-ocr-service")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="DeepSeek-OCR PII Validation Service (Transformers-only)")

# Open up CORS for now; restrict to your IPs/domains in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: lock this down
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Globals (model & tokenizer)
# ----------------------------
_model = None
_tokenizer = None
_device = "cpu"
_dtype = torch.float32


class ValidationRequest(BaseModel):
    image_base64: str
    prompt: str
    category: str
    temperature: float = 0.0
    max_tokens: int = 512


class ValidationResponse(BaseModel):
    text: str
    output: Optional[str] = None
    latency_ms: int
    cached: bool = False


def _pick_device_and_dtype():
    """
    Decide device and dtype.
    - If CUDA is available:
        - If compute capability <= 7.x (e.g., T4=7.5) -> fp16
        - Else (A10/A100/L4 and newer) -> bfloat16
    - Else CPU -> float32
    """
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            if major <= 7:
                return "cuda", torch.float16
            else:
                return "cuda", torch.bfloat16
        except Exception:
            return "cuda", torch.float16
    return "cpu", torch.float32


def _load_tokenizer(model_id: str):
    """
    Load slow tokenizer. Prefer AutoTokenizer(use_fast=False), fall back to LlamaTokenizer if needed.
    """
    try:
        tok = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,   # critical: force slow tokenizer
        )
        return tok
    except Exception as e:
        logger.warning("AutoTokenizer slow failed: %s; falling back to LlamaTokenizer(use_fast=False)", e)
        # LlamaTokenizer (slow) requires sentencepiece installed
        tok = LlamaTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        return tok


def _load_model_and_tokenizer():
    global _model, _tokenizer, _device, _dtype

    logger.info("Loading model (Transformers-only)...")
    logger.info("Transformers version: %s", HF_VER)

    _device, _dtype = _pick_device_and_dtype()
    logger.info("Loading model on device=%s with dtype=%s", _device, _dtype)

    model_id = os.environ.get("DEEPSEEK_OCR_MODEL", "deepseek-ai/DeepSeek-OCR")

    # Tokenizer (slow)
    _tokenizer_local = _load_tokenizer(model_id)

    # Model
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=_dtype,
        low_cpu_mem_usage=True,
        # revision=os.environ.get("DEEPSEEK_OCR_REVISION", None),  # optionally pin commit
    )
    model = model.eval().to(_device)

    return model, _tokenizer_local


@app.on_event("startup")
def _startup():
    global _model, _tokenizer
    try:
        _model, _tokenizer = _load_model_and_tokenizer()
        logger.info("Model and tokenizer loaded successfully.")
    except Exception as e:
        logger.error("Failed to load model", exc_info=True)
        raise


@app.get("/healthz")
def health_check():
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Minimal device info
    info = {"status": "healthy", "transformers": HF_VER, "device": _device, "dtype": str(_dtype)}
    if torch.cuda.is_available():
        info["cuda_name"] = torch.cuda.get_device_name(0)
    return info


def _decode_image_b64(image_b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")


@app.post("/validate_piicrop", response_model=ValidationResponse)
def validate_pii_crop(request: ValidationRequest):
    """
    Validate a PII candidate by OCRing a cropped image region with DeepSeek-OCR.

    - The prompt should request faithful extraction (or a structured JSON verdict if you prefer).
    - This endpoint returns the raw OCR text in 'text'/'output'.
    """
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        image = _decode_image_b64(request.image_base64)

        # DeepSeek-OCR exposes an .infer(...) method via trust_remote_code
        # We keep conservative defaults for image sizes; adjust if needed.
        with torch.no_grad():
            res = _model.infer(
                _tokenizer,
                prompt=request.prompt,
                image_file=image,      # PIL Image is accepted by DeepSeek-OCR trust_remote_code
                output_path=None,
                base_size=1024,        # internal long-side base size
                image_size=640,        # model's processing size
                crop_mode=True,        # we're sending a crop
                save_results=False,
                test_compress=True,    # let model try compressed representation for speed
                temperature=request.temperature,
                max_new_tokens=request.max_tokens,
            )

        # DeepSeek-OCR typically returns a dict with 'text' or 'output'
        text = res.get("text", None)
        if text is None:
            text = res.get("output", "")

        latency_ms = int((time.time() - start) * 1000)
        return ValidationResponse(text=text, output=text, latency_ms=latency_ms, cached=False)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Validation error", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")


if __name__ == "__main__":
    import uvicorn
    # Example run: uvicorn deepseek_ocr_service_example:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
