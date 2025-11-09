"""
Example DeepSeek-OCR FastAPI Service for Remote GPU Deployment

This is a reference implementation of the remote GPU service that the pipeline
calls for L3 PII validation.

Deploy this on a GPU instance (A10/A100/L4) and expose via HTTPS.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
from PIL import Image
import json
import logging
from typing import Optional

# DeepSeek-OCR imports (adjust based on your deployment method)
try:
    # Option 1: vLLM deployment
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
    USE_VLLM = True
except ImportError:
    try:
        # Option 2: Transformers deployment
        from transformers import AutoModel, AutoTokenizer
        import torch
        USE_VLLM = False
        USE_TRANSFORMERS = True
    except ImportError:
        USE_VLLM = False
        USE_TRANSFORMERS = False
        logging.warning("No DeepSeek-OCR backend available - install vLLM or transformers")

app = FastAPI(title="DeepSeek-OCR PII Validation Service")

# CORS middleware (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your pipeline IPs in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (load once at startup)
_model = None
_tokenizer = None


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


@app.on_event("startup")
async def load_model():
    """Load DeepSeek-OCR model at startup."""
    global _model, _tokenizer
    
    if not (USE_VLLM or USE_TRANSFORMERS):
        logging.error("No backend available - service will return errors")
        return
    
    try:
        if USE_VLLM:
            logging.info("Loading DeepSeek-OCR via vLLM...")
            _model = LLM(
                model="deepseek-ai/DeepSeek-OCR",
                enable_prefix_caching=False,
                mm_processor_cache_gb=0,
                logits_processors=[NGramPerReqLogitsProcessor]
            )
            logging.info("Model loaded successfully")
        
        elif USE_TRANSFORMERS:
            logging.info("Loading DeepSeek-OCR via Transformers...")
            _tokenizer = AutoTokenizer.from_pretrained(
                "deepseek-ai/DeepSeek-OCR",
                trust_remote_code=True
            )
            _model = AutoModel.from_pretrained(
                "deepseek-ai/DeepSeek-OCR",
                _attn_implementation='flash_attention_2',
                trust_remote_code=True,
                use_safetensors=True
            )
            _model = _model.eval().cuda().to(torch.bfloat16)
            logging.info("Model loaded successfully")
    
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    if _model is None and not USE_TRANSFORMERS:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "backend": "vLLM" if USE_VLLM else "transformers"}


@app.post("/validate_piicrop", response_model=ValidationResponse)
async def validate_pii_crop(request: ValidationRequest):
    """
    Validate PII candidate using DeepSeek-OCR.
    
    Args:
        request: Validation request with image and prompt
    
    Returns:
        Validation response with JSON verdict
    """
    import time
    start_time = time.time()
    
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode image
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Prepare input based on backend
        if USE_VLLM:
            # vLLM format
            model_input = [{
                "prompt": request.prompt,
                "multi_modal_data": {"image": image}
            }]
            
            sampling_param = SamplingParams(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                extra_args=dict(
                    ngram_size=30,
                    window_size=90,
                    whitelist_token_ids={128821, 128822},  # <td>, </td>
                ),
                skip_special_tokens=False,
            )
            
            # Generate
            outputs = _model.generate(model_input, sampling_param)
            text = outputs[0].outputs[0].text
        
        elif USE_TRANSFORMERS:
            # Transformers format
            res = _model.infer(
                _tokenizer,
                prompt=request.prompt,
                image_file=image,  # Can pass PIL Image directly
                output_path=None,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=True
            )
            text = res.get("text", res.get("output", ""))
        
        else:
            raise HTTPException(status_code=503, detail="No backend available")
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return ValidationResponse(
            text=text,
            output=text,
            latency_ms=latency_ms,
            cached=False
        )
    
    except Exception as e:
        logging.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

