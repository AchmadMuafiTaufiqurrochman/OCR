from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging
import os
import gc
import torch
from onnxtr.io import DocumentFile
from services.OCR import load_ocr_model  # Import model yang sudah dimuat di docktr.py
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


# Logging alakadarnya
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Load ENV gc treshold
        load_dotenv()
        pythongcthrd = os.getenv("PYTHONGCTHRD")
        gc.set_threshold(*map(int, pythongcthrd.split(",")))

        # Konfigurasi PyTorch
        torch.set_float32_matmul_precision('high')
        logger.info("Loading OCR model...")
        with torch.inference_mode():
            app.state.ocr_model = load_ocr_model()  # Gunakan model yang sudah dimuat di docktr.py
            app.state.ocr_model.det_predictor.model.postprocessor.bin_thresh = 0.2
            app.state.ocr_model.det_predictor.model.postprocessor.box_thresh = 0.1
        logger.info("OCR model loaded successfully.")
        yield
    finally:
        del app.state.ocr_model
        gc.collect()
        logger.info("Resources cleaned up.")

app = FastAPI(lifespan=lifespan)

# default limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

# deklarasi rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root(request: Request):
    return {"Hello": "World"}

@app.post("/upload")
@limiter.limit("5/minute")
async def upload(request: Request, file: UploadFile = File(...), ):
    try:
        file_bytes = await file.read()
        await file.close()

        doc = DocumentFile.from_images([file_bytes])
        del file_bytes

        result = await asyncio.to_thread(app.state.ocr_model, doc)
        del doc

        if isinstance(result, dict) and "error" in result:
            error = result.get("error", "Unknown error")
            del result
            logger.error(f"Processing error: {error}")
            return JSONResponse({"error": error}, status_code=500)

        output = result.render().split("\n")
        del result
        gc.collect()
        return JSONResponse({
            "status": "success",
            "data": output
        })

    except Exception as e:
        logger.exception("Error processing file")
        del file
        gc.collect()
        return JSONResponse({"error": str(e)}, status_code=500)
