from fastapi import FastAPI, HTTPException, UploadFile, File
import shutil
import os
from services.OCR import process_ocr

app = FastAPI()
TEMP_FOLDER = "public/images"

@app.get("/")
def root():
    return {"message": "uwu"}

@app.post("/ktp")
def run_ocr(file: UploadFile = File(...)):
    try:
        temp_file_path = os.path.join(TEMP_FOLDER, file.filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Proses OCR
        result = process_ocr(temp_file_path)
        
        # Hapus file setelah diproses
        os.remove(temp_file_path)
        
        return {"status": "success", "data": result}
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File tidak ditemukan")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
