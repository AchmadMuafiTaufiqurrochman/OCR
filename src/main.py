from fastapi import FastAPI, HTTPException, UploadFile, File
import os
import aiofiles
import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from services.OCR import init_model, process_ocr

TEMP_FOLDER = "public/images"
executor = ThreadPoolExecutor(max_workers=2)  # Batasi jumlah worker agar lebih efisien
model = None  # Model diinisialisasi dalam lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle FastAPI untuk inisialisasi dan shutdown model."""
    global model
    print("Loading OCR Model...")
    model = init_model()  # Load model sekali di awal aplikasi
    yield
    print("Shutting down OCR Model...")
    executor.shutdown(wait=True)  # Tutup thread pool untuk mencegah kebocoran memori
    del model  # Hapus referensi ke model
    gc.collect()  # Paksa garbage collection

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "uwu"}

@app.post("/ktp")
async def run_ocr(file: UploadFile = File(...)):
    """API untuk menjalankan OCR pada gambar yang diunggah."""
    try:
        temp_file_path = os.path.join(TEMP_FOLDER, file.filename)

        # Simpan file secara asynchronous
        async with aiofiles.open(temp_file_path, "wb") as buffer:
            content = await file.read()  # Membaca seluruh file ke dalam memori
            await buffer.write(content)  # Menulis seluruh file ke disk

        # Proses OCR secara async di thread pool agar tidak blocking
        result = await asyncio.get_running_loop().run_in_executor(
            executor, lambda: process_ocr(temp_file_path, model)
        )

        # Hapus file setelah diproses dengan pengecekan
        try:
            os.remove(temp_file_path)
        except Exception as e:
            print(f"Error saat menghapus file: {e}")

        return {
            "status": "success",
            "data": result  # Hasil dalam bentuk list agar JSON lebih rapi
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File tidak ditemukan")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
