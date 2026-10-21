from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import UnidentifiedImageError

from src import config
from src.predict import load_model, predict_image

app = FastAPI(title="Skin Lesion Classifier (research demo)")
model = load_model()


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Please upload a JPEG or PNG image.")
    data = await file.read()
    try:
        return predict_image(model, data)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="That file could not be read as an image.")