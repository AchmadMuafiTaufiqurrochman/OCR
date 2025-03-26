import os
import gc
import torch
from dotenv import load_dotenv
from onnxtr.models import ocr_predictor, EngineConfig


# Load environment variables
load_dotenv()
DET_ARCH = os.getenv("DET_ARCH", "db_resnet50")
RECO_ARCH = os.getenv("RECO_ARCH", "parseq")
DET_BS = int(os.getenv("DETECTION_BATCH_SIZE", "1"))
RECO_BS = int(os.getenv("RECO_BATCH_SIZE", "16"))

# Konfigurasi PyTorch
torch.set_float32_matmul_precision('high')

# Fungsi untuk load model
def load_ocr_model():
    with torch.inference_mode():
        model = ocr_predictor(
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
        model.det_predictor.model.postprocessor.bin_thresh = 0.2
        model.det_predictor.model.postprocessor.box_thresh = 0.1
    return model

# Load model saat modul diimpor
ocr_model = load_ocr_model()
