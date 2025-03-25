from onnxtr.io import DocumentFile
from onnxtr.models import ocr_predictor, EngineConfig
import os
import gc

def load_model():
    """Inisialisasi model setiap kali dipanggil agar tidak menumpuk di memori."""
    model = ocr_predictor(
        det_arch='db_resnet50',
        reco_arch='parseq',
        assume_straight_pages=True,
        straighten_pages=False,
        export_as_straight_boxes=False,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
        detect_orientation=False,
        detect_language=False,
        disable_crop_orientation=False,
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

def process_ocr(image_path: str):
    """Proses OCR dengan model yang baru setiap request."""
    if not os.path.exists(image_path):
        raise FileNotFoundError("File tidak ditemukan")

    model = load_model()  # Load model setiap request (menghindari memory leak)
    input_page = DocumentFile.from_images(image_path)
    result = model(input_page)

    formatted_result = result.render().split("\n")

    # Paksa hapus model dari memori setelah digunakan
    del model
    del result
    gc.collect()

    return formatted_result  # Kembalikan list agar JSON lebih rapi
