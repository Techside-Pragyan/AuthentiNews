from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import io
from PyPDF2 import PdfReader
import docx

from api.models.classifier import get_classifier
from api.services.scraper import extract_article_details
from utils.preprocessing import clean_text, get_suspicious_phrases

app = FastAPI(title="AuthentiNews API", description="AI-powered Fake News Detection System")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

class URLRequest(BaseModel):
    url: str

@app.get("/")
async def root():
    return {"message": "AuthentiNews API is running"}

@app.post("/predict/text")
async def predict_text(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    classifier = get_classifier()
    prediction = classifier.predict(request.text)
    
    # Add explainability (Suspicious phrases)
    prediction["suspicious_phrases"] = get_suspicious_phrases(request.text)
    
    return prediction

@app.post("/predict/url")
async def predict_url(request: URLRequest):
    article = extract_article_details(request.url)
    if not article:
        raise HTTPException(status_code=400, detail="Could not extract content from URL")
    
    classifier = get_classifier()
    prediction = classifier.predict(article["text"])
    
    prediction["article_info"] = {
        "title": article["title"],
        "authors": article["authors"],
        "keywords": article["keywords"]
    }
    prediction["suspicious_phrases"] = get_suspicious_phrases(article["text"])
    
    return prediction

@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    content = ""
    extension = file.filename.split(".")[-1].lower()
    
    file_bytes = await file.read()
    
    if extension == "txt":
        content = file_bytes.decode("utf-8")
    elif extension == "pdf":
        pdf = PdfReader(io.BytesIO(file_bytes))
        for page in pdf.pages:
            content += page.extract_text()
    elif extension == "docx":
        doc = docx.Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            content += para.text + "\n"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    if not content.strip():
        raise HTTPException(status_code=400, detail="File is empty or could not be read")
    
    classifier = get_classifier()
    prediction = classifier.predict(content)
    prediction["suspicious_phrases"] = get_suspicious_phrases(content)
    
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
