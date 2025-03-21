from onnxtr.io import DocumentFile
from onnxtr.models import ocr_predictor, EngineConfig
import os
def init_model():
    model = ocr_predictor(
    det_arch='db_resnet50', 
    reco_arch='parseq', 
    # pretrained=True,
    # Document related parameters
    assume_straight_pages=True,  # set to `False` if the pages are not straight (rotation, perspective, etc.) (default: True)
    straighten_pages=False,  # set to `True` if the pages should be straightened before final processing (default: False)
    export_as_straight_boxes=False,  # set to `True` if the boxes should be exported as if the pages were straight (default: False)
    # Preprocessing related parameters
    preserve_aspect_ratio=True,  # set to `False` if the aspect ratio should not be preserved (default: True)
    symmetric_pad=True,  # set to `False` to disable symmetric padding (default: True)
    # Additional parameters - meta information
    detect_orientation=False,  # set to `True` if the orientation of the pages should be detected (default: False)
    detect_language=False, # set to `True` if the language of the pages should be detected (default: False)
    # Orientation specific parameters in combination with `assume_straight_pages=False` and/or `straighten_pages=True`
    disable_crop_orientation=False,  # set to `True` if the crop orientation classification should be disabled (default: False)
    disable_page_orientation=True,  # set to `True` if the general page orientation classification should be disabled (default: False)
    # DocumentBuilder specific parameters
    resolve_lines=True,  # whether words should be automatically grouped into lines (default: True)
    resolve_blocks=False,  # whether lines should be automatically grouped into blocks (default: False)
    paragraph_break=0.035,  # relative length of the minimum space separating paragraphs (default: 0.035)
    # OnnxTR specific parameters
    load_in_8_bit=False,  # set to `True` to load 8-bit quantized models instead of the full precision onces (default: False)
    # Advanced engine configuration options
    det_engine_cfg=EngineConfig(),  # detection model engine configuration (default: internal predefined configuration)
    reco_engine_cfg=EngineConfig(),  # recognition model engine configuration (default: internal predefined configuration)
    clf_engine_cfg=EngineConfig(),  # classification (orientation) model engine configuration (default: internal predefined configuration)
    )
    model.det_predictor.model.postprocessor.bin_thresh = 0.2
    model.det_predictor.model.postprocessor.box_thresh = 0.1
    return model

model = init_model()

def process_ocr(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError("File tidak ditemukan")
    
    input_page = DocumentFile.from_images(image_path)
    result = model(input_page)
    return result.render()