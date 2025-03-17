from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Memilih model OCR
model = ocr_predictor(det_arch='db_resnet50', reco_arch='parseq', pretrained=True)

# Threshold
model.det_predictor.model.postprocessor.bin_thresh = 0.2
model.det_predictor.model.postprocessor.box_thresh = 0.1

# Fungsi untuk memproses gambar dengan OCR
def process_ocr(image_path):
    # Baca gambar
    input_page = DocumentFile.from_images(image_path)
    
    # Proses OCR
    result = model(input_page)
    
    # Ambil output dalam format JSON
    text_output = result.export()
    
    return text_output  # Return Hasil OCR
