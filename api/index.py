from fastapi import FastAPI, UploadFile, File
import requests
import os

app = FastAPI()

TOKEN = os.getenv("HF_TOKEN")
# SigLIP2 is the 2026 state-of-the-art for face-based forgery
API_URL = "https://router.huggingface.co/hf-inference/models/prithivMLmods/Deepfake-Detect-Siglip2"

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        headers = {
            "Authorization": f"Bearer {TOKEN.strip()}",
            "Content-Type": "image/jpeg"
        }
        
        response = requests.post(API_URL, headers=headers, data=contents, timeout=25)
        
        if response.status_code == 200:
            return response.json()
        
        # Mask specific API errors from the end user
        return {"error": "Processing error. Please try again later."}
    
    except Exception:
        return {"error": "Server is currently busy."}
