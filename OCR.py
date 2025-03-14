# LETSGOO IMPORTT
from doctr.io import DocumentFile
import matplotlib.pyplot as plt
import numpy as np
from doctr.models import ocr_predictor
import json

# Memilih model
model = ocr_predictor(det_arch='db_resnet50', reco_arch='parseq', pretrained=True)

# Treshold
model.det_predictor.model.postprocessor.bin_thresh = 0.2
model.det_predictor.model.postprocessor.box_thresh = 0.1

# Input gambar
input_page = DocumentFile.from_images("images\WhatsApp Image 2025-03-12 at 16.27.58_4c841a6d.jpg")

# Processing
result = model(input_page)

# mengambil Output
text_output = result.export()

# path export json
file_path = "Ini_hasil_JSON/data.json"

# Export json
with open(file_path, 'w') as f:
    json.dump(text_output, f, indent=4)

# Cek kondisi Apakah json sudah di ekstraksi
if text_output:
  print("Json telah di ekstraksi")