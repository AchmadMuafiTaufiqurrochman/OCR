import os
from dotenv import load_dotenv
from onnxtr.models import ocr_predictor, EngineConfig


# Load environment variables
load_dotenv()
DET_ARCH = os.getenv("DET_ARCH")
RECO_ARCH = os.getenv("RECO_ARCH")
DET_BS = int(os.getenv("DETECTION_BATCH_SIZE"))
RECO_BS = int(os.getenv("RECO_BATCH_SIZE"))

# Fungsi untuk load model
def load_ocr_model():
    return ocr_predictor(
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