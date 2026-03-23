from fastapi import FastAPI, UploadFile, File
import requests
import os

app = FastAPI()

TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/umm-maybe/AI-image-detector"

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "image/jpeg"
        }
        
        response = requests.post(API_URL, headers=headers, data=contents)
        
        if response.status_code == 200:
            return response.json()
        return {"error": "Something went wrong. Please try again later."}
    
    except Exception as e:
        return {"error": "Something went wrong."}
