# AuthentiNews: AI-Powered Fake News Detection System

AuthentiNews is a full-stack application designed to combat misinformation by analyzing news articles using state-of-the-art Transformer models. It provides real-time classification (Real vs. Fake) with confidence scores and linguistic insights.

## 🚀 Key Features
- **Multi-Input Support**: Analyze text, URLs, and uploaded documents (PDF, DOCX, TXT).
- **Advanced AI**: Powered by DistilBERT for high-accuracy text classification.
- **Explainability**: Highlights suspicious phrases and providing confidence-based verdicts.
- **Premium UI**: Modern, glassmorphism-inspired dashboard for a seamless user experience.
- **Real-Time Analysis**: Instant results powered by a FastAPI backend.

## 🛠️ Technology Stack
- **Backend**: FastAPI, Transformers (Hugging Face), PyTorch, Newspaper3k (Scraping), PyPDF2, python-docx.
- **Frontend**: HTML5, Vanilla CSS, Javascript (ES6+).
- **Model**: `therealcyberlord/fake-news-classification-distilbert` (DistilBERT fine-tuned for misinformation).

## 📂 Project Structure
- `/api`: FastAPI backend and services.
- `/data`: Scripts for downloading and preparing datasets.
- `/models`: Training scripts for fine-tuning custom models.
- `/frontend`: Responsive user interface.
- `/utils`: Preprocessing and NLP utility functions.

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.8+
- Node.js (Optional, if using a dev server for frontend)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Backend API
```bash
python -m uvicorn api.main:app --reload
```
The API will be available at `http://localhost:8000`.

### 4. Run the Frontend
Opening `frontend/index.html` in any modern web browser will launch the application.

## 🧠 Model Training (Optional)
To train your own model on the provided dataset:
1. Run `python data/prepare_data.py` to download the news dataset.
2. Run `python models/train_model.py` to start fine-tuning (GPU recommended).

## ⚠️ Disclaimer
Fake news detection is inherently complex. This tool is designed to assist in verification and should not be used as the sole source of truth. Always cross-reference with multiple verified sources.

---
Built with ❤️ by Antigravity