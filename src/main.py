from fastapi import FastAPI, HTTPException, UploadFile, File
import os
import aiofiles
from contextlib import asynccontextmanager
from services.OCR import process_ocr, load_model

TEMP_FOLDER = "public/images"

# Inisialisasi model saat aplikasi startup
ocr_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle FastAPI untuk inisialisasi dan shutdown service."""
    global ocr_model
    ocr_model = load_model()  # Model hanya dimuat sekali saat startup
    yield
    del ocr_model  # Hapus model dari memori saat aplikasi shutdown

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "uwu"}

@app.post("/ktp")
async def run_ocr(file: UploadFile = File(...)):
    """API untuk menjalankan OCR pada gambar yang diunggah."""
    temp_file_path = os.path.join(TEMP_FOLDER, file.filename)

    try:
        # Simpan file secara asynchronous
        async with aiofiles.open(temp_file_path, "wb") as buffer:
            content = await file.read()
            await buffer.write(content)

        # Jalankan OCR langsung, tanpa ThreadPoolExecutor
        result = process_ocr(temp_file_path, ocr_model)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File tidak ditemukan")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Pastikan file selalu dihapus setelah digunakan
        try:
            os.remove(temp_file_path)
        except Exception as e:
            print(f"Error saat menghapus file: {e}")

    return {
        "status": "success",
        "data": result
    }
