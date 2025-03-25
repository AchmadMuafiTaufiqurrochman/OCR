from fastapi import FastAPI, HTTPException, UploadFile, File
import os
import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from services.OCR import init_model, process_ocr

TEMP_FOLDER = "public/images"
executor = ThreadPoolExecutor()
model = None  # Model diinisialisasi dalam lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Loading OCR Model...")
    model = init_model()  # Load model sekali di awal aplikasi
    yield
    print("Shutting down OCR Model...")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "uwu"}

@app.post("/ktp")
async def run_ocr(file: UploadFile = File(...)):
    try:
        temp_file_path = os.path.join(TEMP_FOLDER, file.filename)

        # Simpan file secara asynchronous
        async with aiofiles.open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(1024):
                await buffer.write(chunk)

        # Proses OCR secara async di thread pool agar tidak blocking
        result = await asyncio.get_running_loop().run_in_executor(
            executor, lambda: process_ocr(temp_file_path, model)
        )

        # Hapus file setelah diproses
        os.remove(temp_file_path)

        return {
            "status": "success",
            "data": result  # Hasil dalam bentuk list agar JSON lebih rapi
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File tidak ditemukan")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))