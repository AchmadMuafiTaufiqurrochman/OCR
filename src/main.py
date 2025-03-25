from fastapi import FastAPI, HTTPException, UploadFile, File
import os
import aiofiles
import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from services.OCR import process_ocr

TEMP_FOLDER = "public/images"
executor = ThreadPoolExecutor(max_workers=1)  # Batasi jumlah worker

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle FastAPI untuk inisialisasi dan shutdown service."""
    yield
    executor.shutdown(wait=True)  # Pastikan thread pool dibersihkan setelah digunakan
    gc.collect()  # Paksa garbage collection

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

        # Proses OCR secara async di thread pool agar tidak blocking
        result = await asyncio.get_running_loop().run_in_executor(
            executor, lambda: process_ocr(temp_file_path)
        )

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

        # Paksa garbage collection
        gc.collect()

    return {
        "status": "success",
        "data": result  # JSON lebih rapi dalam bentuk list
    }
