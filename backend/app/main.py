from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import io
import json
import numpy as np

from .models.classifier import DefectClassifier
from .models.segmenter import Segmenter
from .utils.gradcam import generate_gradcam
from .models.root_cause_bert import RootCauseClassifier
from .models.clip_fusion import CLIPFusion


# Initialize models
classifier = DefectClassifier()
segmenter = Segmenter()
root_cause_clf = RootCauseClassifier()
clip_fusion = CLIPFusion()

app = FastAPI()

# Mount static files (frontend)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
# Initialize models (lazy loading)
classifier = DefectClassifier()
segmenter = Segmenter()

@app.get("/")
async def read_root():
    return FileResponse("app/static/index.html")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), text: str = Form("")):
    # Read image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Defect classification
    defect_result = classifier.predict(image)

    # Segmentation (if available)
    mask = segmenter.segment(image)

    # Root cause analysis
    root_cause_result = root_cause_clf.predict(text)

    # CLIP fusion
    similarity = clip_fusion.fuse(image, text)

    return JSONResponse({
        "defect": defect_result,
        "mask": mask,  # can be large; you may want to compress later
        "root_cause": root_cause_result,
        "clip_similarity": similarity
    })

# Optional separate endpoints for testing
@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    result = classifier.predict(image)
    return JSONResponse(result)

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    mask = segmenter.segment(image)
    return JSONResponse({"mask": mask})

