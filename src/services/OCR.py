from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os

def init_model():
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='parseq', pretrained=True)
    model.det_predictor.model.postprocessor.bin_thresh = 0.2
    model.det_predictor.model.postprocessor.box_thresh = 0.1
    return model

model = init_model()

def process_ocr(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError("File tidak ditemukan")
    
    input_page = DocumentFile.from_images(image_path)
    result = model(input_page)
    return result.export()