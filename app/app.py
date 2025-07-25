from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model and vectorizer
with open(r"D:\FastAPI\Fake_News_Project\model\model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"D:\FastAPI\Fake_News_Project\model\vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Define FastAPI app
app = FastAPI(title="📰 Fake News Detection API")

# Request structure
class NewsItem(BaseModel):
    content: str

# Route: Prediction endpoint
@app.post("/predict/")
def predict_news(item: NewsItem):
    text_vector = vectorizer.transform([item.content])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0].max()

    return {
        "verdict": "Real" if prediction == 1 else "Fake",
        "confidence": round(probability * 100, 2)
    }
