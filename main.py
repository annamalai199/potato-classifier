from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
import tempfile
import os
import logging
import subprocess
from uuid import uuid4
from typing import List

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# App & CORS
# -------------------------
app = FastAPI(title="Potato Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load model
# -------------------------
MODEL_PATH = "saved_models/1.keras"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

MODEL = None
if os.path.exists(MODEL_PATH):
    logger.info(f"Loading model from {MODEL_PATH} ...")
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    logger.info("✅ Model loaded successfully")
else:
    logger.error(f"❌ Model not found at {MODEL_PATH}")

# -------------------------
# Health check
# -------------------------
@app.get("/")
def home():
    return {"message": "Server is running"}

@app.get("/ping")
def ping():
    return {"message": "Hello, I am alive"}

# -------------------------
# Helpers
# -------------------------
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))
    return np.array(image)

def reencode_video(input_path: str) -> str:
    output_path = os.path.join(tempfile.gettempdir(), f"{uuid4().hex}_final.mp4")
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Re-encoded video saved at {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"❌ FFmpeg error: {e}")
        return input_path

# -------------------------
# Image prediction
# -------------------------
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    logger.info(f"Received image: {file.filename}")
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    logger.info(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}")
    return {"prediction": predicted_class, "confidence": confidence}

# -------------------------
# Video prediction
# -------------------------
@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    logger.info(f"Received video: {file.filename}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 20

    raw_out_path = os.path.join(tempfile.gettempdir(), f"{uuid4().hex}_raw.mp4")
    out = cv2.VideoWriter(raw_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_predictions: List[str] = []
    frame_confidences: List[float] = []
    frame_interval = 15
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_interval != 0:
            out.write(frame)
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img_batch = np.expand_dims(img, 0)

        predictions = MODEL.predict(img_batch, verbose=0)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        frame_predictions.append(predicted_class)
        frame_confidences.append(confidence)

        cv2.putText(frame, f"{predicted_class} ({confidence:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

    cap.release()
    out.release()
    os.remove(video_path)

    final_out_path = reencode_video(raw_out_path)

    if frame_predictions:
        final_class = max(set(frame_predictions), key=frame_predictions.count)
        avg_confidence = float(np.mean(frame_confidences))
    else:
        final_class, avg_confidence = "No frames processed", 0.0

    video_url = f"/download/{os.path.basename(final_out_path)}"
    logger.info(f"Processed video available at: {video_url}")

    return {"final_prediction": final_class, "average_confidence": avg_confidence, "video_url": video_url}

# -------------------------
# Download endpoint
# -------------------------
@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

# -------------------------
# Handle 404
# -------------------------
@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse(status_code=404, content={"message": "Endpoint not found"})

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), log_level="info")
