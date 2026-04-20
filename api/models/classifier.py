from transformers import pipeline
import torch
import numpy as np

class NewsClassifier:
    def __init__(self):
        # We use a pre-trained model for the production service
        # If the user has trained their own, they can update this path
        model_id = "therealcyberlord/fake-news-classification-distilbert"
        print(f"Loading model: {model_id}...")
        self.classifier = pipeline(
            "text-classification", 
            model=model_id, 
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1
        )
        print("Model loaded.")

    def predict(self, text):
        # Truncate text to fit model context (512 tokens approx)
        truncated_text = text[:2000] 
        results = self.classifier(truncated_text)[0]
        
        # Format the results
        # Usually labels are 'LABEL_0' (Fake) and 'LABEL_1' (Real) or similar
        # Based on this specific model's config, LABEL_0 is typically FAKE, LABEL_1 is REAL
        fake_score = results[0]['score']
        real_score = results[1]['score']
        
        prediction = "REAL" if real_score > fake_score else "FAKE"
        confidence = real_score if prediction == "REAL" else fake_score
        
        return {
            "prediction": prediction,
            "confidence": round(float(confidence) * 100, 2),
            "probabilities": {
                "real": round(float(real_score) * 100, 2),
                "fake": round(float(fake_score) * 100, 2)
            }
        }

# Singleton instance
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = NewsClassifier()
    return _classifier
