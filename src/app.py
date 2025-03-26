from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from onnxtr.io import DocumentFile
from onnxtr.models import ocr_predictor, EngineConfig
import torch
import os
import gc
import asyncio
import logging

gc.set_threshold(500, 10, 5)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
DET_ARCH = os.getenv("DET_ARCH", "db_resnet50")
RECO_ARCH = os.getenv("RECO_ARCH", "parseq")
DET_BS = int(os.getenv("DETECTION_BATCH_SIZE", "1"))
RECO_BS = int(os.getenv("RECO_BATCH_SIZE", "16"))

torch.set_float32_matmul_precision('high')

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with torch.inference_mode():
            app.state.ocr_model = ocr_predictor(
				det_arch=DET_ARCH,
				reco_arch=RECO_ARCH,
				det_bs=DET_BS,
				reco_bs=RECO_BS,
				assume_straight_pages=True,
				straighten_pages=False,
				export_as_straight_boxes=False,
				preserve_aspect_ratio=True,
				symmetric_pad=True,
				detect_orientation=False,
				detect_language=False,
				disable_crop_orientation=True,
				disable_page_orientation=True,
				resolve_lines=True,
				resolve_blocks=False,
				paragraph_break=0.035,
				load_in_8_bit=False,
				det_engine_cfg=EngineConfig(),
				reco_engine_cfg=EngineConfig(),
				clf_engine_cfg=EngineConfig(),
			)
            app.state.ocr_model.det_predictor.model.postprocessor.bin_thresh = 0.2
            app.state.ocr_model.det_predictor.model.postprocessor.box_thresh = 0.1
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