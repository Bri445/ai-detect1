from fastapi import FastAPI, UploadFile, File
import requests
import os

app = FastAPI()

# Configuration
TOKEN = os.getenv("HF_TOKEN")
# Specialized 2026 SigLIP2 model for Deepfake detection
API_URL = "https://router.huggingface.co/hf-inference/models/prithivMLmods/Deepfake-Detect-Siglip2"

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "image/jpeg"
        }
        
        # Calling the Hugging Face Router
        response = requests.post(API_URL, headers=headers, data=contents, timeout=20)
        
        if response.status_code == 200:
            results = response.json()
            # This model returns 'Fake' and 'Real' labels
            return results
        
        print(f"API ERROR: {response.status_code} - {response.text}")
        return {"error": "Deepfake analysis failed. Please try again."}
    
    except Exception as e:
        print(f"SERVER ERROR: {str(e)}")
        return {"error": "Internal Server Error."}
