from fastapi import FastAPI, File, UploadFile
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
from typing import List
import logging

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------
# Initialize app
# -------------------------
app = FastAPI(title="Potato Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load model safely
# -------------------------
MODEL = None
MODEL_PATH = "saved_models/1.keras"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

if os.path.exists(MODEL_PATH):
    logging.info(f"Loading model from {MODEL_PATH} ...")
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    logging.info("✅ Model loaded successfully")
else:
    logging.error(f"❌ Model file not found at {MODEL_PATH}. Upload it to Railway project!")

# -------------------------
# Health check
# -------------------------
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

# -------------------------
# Helper: read image
# -------------------------
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))
    return np.array(image)

# -------------------------
# Image Prediction Endpoint
# -------------------------
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if MODEL is None:
        return JSONResponse(
            content={"error": "Model not loaded. Please upload 'saved_models/1.keras' to Railway."},
            status_code=500,
        )

    logging.info(f"Received image: {file.filename}")
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    logging.info(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}")
    return {"prediction": predicted_class, "confidence": confidence}

# -------------------------
# Video Prediction Endpoint
# -------------------------
@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    if MODEL is None:
        return JSONResponse(
            content={"error": "Model not loaded. Please upload 'saved_models/1.keras' to Railway."},
            status_code=500,
        )

    logging.info(f"Received video: {file.filename}")

    # Save video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name

    cap = cv2.VideoCapture(video_path)

    # Prepare output video
    out_path = os.path.join(
        tempfile.gettempdir(),
        os.path.basename(video_path).replace(".mp4", "_predicted.mp4")
    )

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

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

        # Prepare frame for prediction
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img_batch = np.expand_dims(img, 0)

        predictions = MODEL.predict(img_batch, verbose=0)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        frame_predictions.append(predicted_class)
        frame_confidences.append(confidence)

        # Annotate frame
        cv2.putText(frame, f"{predicted_class} ({confidence:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0), 3, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()
    os.remove(video_path)

    if frame_predictions:
        final_class = max(set(frame_predictions), key=frame_predictions.count)
        avg_confidence = float(np.mean(frame_confidences))
    else:
        final_class, avg_confidence = "No frames processed", 0.0

    video_url = f"/download/{os.path.basename(out_path)}"
    logging.info(f"Processed video available at: {video_url}")

    return {
        "final_prediction": final_class,
        "average_confidence": avg_confidence,
        "video_url": video_url
    }

# -------------------------
# Serve Annotated Video
# -------------------------
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    logging.info(f"Serving video: {file_path}")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
