from fastapi import FastAPI, HTTPException
from services.OCR import process_ocr

app = FastAPI()

@app.get("/")
def root():
    return{"uwu"}


@app.get("/ktp")
def run_ocr(image_path: str):
    try:
        result = process_ocr(image_path)
        return {"status": "success", "data": result}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File tidak ditemukan")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))