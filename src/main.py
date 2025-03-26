from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import logging
import os
import gc
from onnxtr.io import DocumentFile
from services.OCR import ocr_model  # Import model yang sudah dimuat di docktr.py

gc.set_threshold(300, 10, 5)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading OCR model...")
        app.state.ocr_model = ocr_model  # Gunakan model yang sudah dimuat di docktr.py
        logger.info("OCR model loaded successfully.")
        yield
    finally:
        del app.state.ocr_model
        gc.collect()
        logger.info("Resources cleaned up.")

app = FastAPI(lifespan=lifespan)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
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

        return output

    except Exception as e:
        logger.exception("Error processing file")
        del file
        gc.collect()
        return JSONResponse({"error": str(e)}, status_code=500)
