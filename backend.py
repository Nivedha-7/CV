import os
import sys
import io
import shutil
import threading
import subprocess
from datetime import datetime
from pathlib import Path
 
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
 
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
 
# =========================
# APP SETUP
# =========================
app = FastAPI(title="Crack Detection API", version="1.0.0")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# =========================
# PATHS — update these to your actual paths
# =========================
MODEL_PATH      = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\best_crack_classifier.pth"
UPLOAD_DIR      = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\uploads"
TRAIN_SCRIPT    = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\train_classifier.py"
 
os.makedirs(UPLOAD_DIR, exist_ok=True)
 
# =========================s
# GLOBAL STATUS TRACKER
# =========================
system_status = {
    "model_loaded": False,
    "training_running": False,
    "training_progress": 0,
    "last_prediction": None,
    "last_upload": None,
    "total_predictions": 0,
    "error": None
}
 
# =========================
# MODEL LOADING
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
label_classes = []
 
def load_model():
    global model, label_classes, system_status
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        label_classes = checkpoint["label_classes"]
 
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(label_classes))
        m.load_state_dict(checkpoint["model_state_dict"])
        m = m.to(device)
        m.eval()
 
        model = m
        system_status["model_loaded"] = True
        system_status["error"] = None
        print(f"✅ Model loaded. Classes: {label_classes}")
    except Exception as e:
        system_status["model_loaded"] = False
        system_status["error"] = str(e)
        print(f"❌ Model load failed: {e}")
 
# Load model on startup
load_model()
 
# =========================
# PREPROCESSING
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
 
# =========================
# HELPER: get_message
# =========================
def get_message(label: str) -> str:
    messages = {
        "Low": (
            "Severity of the crack detected is LOW.\n\n"
            "A minor surface-level crack has been identified. This type of defect does not "
            "currently impact structural integrity or operational performance.\n\n"
            "Recommendation: No immediate action is required. Monitor the affected area "
            "during routine inspections to ensure the crack does not propagate over time."
        ),
        "Medium": (
            "Severity of the crack detected is MEDIUM.\n\n"
            "A noticeable structural crack has been detected, which may worsen if left "
            "unaddressed. This indicates a moderate level of risk.\n\n"
            "Recommendation: Schedule maintenance and perform a detailed inspection within "
            "the next few days to prevent further deterioration."
        ),
        "High": (
            "Severity of the crack detected is HIGH.\n\n"
            "A critical structural defect has been identified. This level of damage may "
            "significantly impact safety, reliability, and overall system performance.\n\n"
            "Recommendation: Immediate attention is required. Repair or replace the affected "
            "component within a week to prevent failure or further damage."
        ),
    }
    return messages.get(label, "Unable to determine crack severity.")
 
# =========================
# ENDPOINT 1: /predict
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image, runs the classification model,
    returns severity + confidence + message.
    """
    if not system_status["model_loaded"] or model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is not loaded. Please check the model path."}
        )
 
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
 
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
 
        predicted_label = label_classes[predicted_idx]
        confidence = round(probabilities[predicted_idx].item() * 100, 2)
 
        prob_dict = {
            label_classes[i]: round(probabilities[i].item() * 100, 2)
            for i in range(len(label_classes))
        }
 
        # Update status tracker
        system_status["last_prediction"] = {
            "filename": file.filename,
            "severity": predicted_label,
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        system_status["total_predictions"] += 1
 
        return {
            "filename": file.filename,
            "severity": predicted_label,
            "confidence": confidence,
            "probabilities": prob_dict,
            "message": get_message(predicted_label)
        }
 
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )
 
# =========================
# ENDPOINT 2: /upload
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Saves uploaded image to the uploads/ folder.
    Returns saved file path.
    """
    try:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
 
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
 
        system_status["last_upload"] = {
            "filename": file.filename,
            "saved_path": save_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
 
        return {
            "success": True,
            "filename": file.filename,
            "saved_path": save_path,
            "message": f"File '{file.filename}' uploaded successfully."
        }
 
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Upload failed: {str(e)}"}
        )
 
# =========================
# ENDPOINT 3: /train
# =========================
def run_training():
    
    global system_status

    print("run_training() started")

    system_status["training_running"] = True
    system_status["training_progress"] = 0
    system_status["error"] = None
 
    try:
        process = subprocess.Popen(
            [sys.executable, TRAIN_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in iter(process.stdout.readline, ''):
            print(line.strip(), flush=True)
        
            if "Epoch" in line and "/" in line:
                try:
                    current = int(current)
                    total = int(total)
                    system_status["training_progress"] = int((current / total) * 100)
                except Exception:
                    pass
        process.wait()
        print("Process finished with code:", process.returncode)

        if process.returncode == 0:
            system_status["training_progress"]=100
            load_model()
            system_status["training_progress"] = 100
            print("Training completed successfully.")
        else:
            system_status["error"] = "Training failed. Check Terminal logs."
            print("Training failed")

    
    except Exception as e:
        system_status["error"] = str(e)
        print("Exception occurred", str(e))
    finally:
        system_status["training_running"] = False
        print("training_running set to False")
        print("Training ended")
 
        
 
@app.post("/train")
async def train(background_tasks: BackgroundTasks):
    print("training_running_status(before check):", system_status["training_running"])  
    if system_status["training_running"]:
        print("Training already in progress - returning 409")
        return JSONResponse(
            status_code=409,
            content={"error": "Training is already running. Check /status for progress."}
        )
 
    # Run in background so API stays responsive
    system_status["error"] = None
    system_status["training_progress"] = 0

    print("Starting training....")


    background_tasks.add_task(run_training)
 
    return {
        "success": True,
        "message": "Training started in background. Use /status to monitor.",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/reset_training")
async def reset_training():
    system_status["training_running"] = False
    system_status["training_progress"] = 0
    system_status["error"] = None
    return {"success": True, "message": "Training status reset."}

# =========================
# ENDPOINT 4: /status
# =========================
@app.get("/status")
async def status():
    return {
        "model_loaded": system_status["model_loaded"],
        "training_running": system_status["training_running"],
        "training_progress": system_status["training_progress"],
        "error": system_status["error"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
 
# =========================
# ROOT
# =========================
@app.get("/")
async def root():
    return {
        "message": "Crack Detection API is running.",
        "endpoints": ["/predict", "/upload", "/train", "/status"]
    }
 