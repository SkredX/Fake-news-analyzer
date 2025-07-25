# Fake-news-analyzer
# Fake-News-Detection-API---FastAPI-ML

## Usage directive:
This Fake News Detection API is a lightweight microservice built using FastAPI that classifies input text as either **Real** or **Fake** news using a pre-trained machine learning model. The classifier is trained using `scikit-learn` and `TF-IDF` vectorization, and the API returns a binary verdict along with a confidence score.

This API can be used to:

- Submit article content for analysis
- Receive a "Real" or "Fake" prediction
- Obtain a confidence score for each prediction
- Integrate into larger workflows, bots, or verification tools

Run the following code in your Python IDE (after basic installations like installing [uv](https://docs.astral.sh/uv/)):

```python
# fake_news_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model and vectorizer
with open(r"model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Create FastAPI app
app = FastAPI(title="Fake News Detection API")

# Request format
class NewsItem(BaseModel):
    content: str

# API route
@app.post("/predict/")
def predict_news(item: NewsItem):
    vec = vectorizer.transform([item.content])
    prediction = model.predict(vec)[0]
    confidence = model.predict_proba(vec)[0].max()

    return {
        "verdict": "Real" if prediction == 1 else "Fake",
        "confidence": round(confidence * 100, 2)
    }
```

Run the following command in your terminal, opened in the working directory:

```bash
uvicorn fake_news_api:app --reload
```
For example, if your file name is saved as fake_news_api.py, use:

```bash
uvicorn fake_news_api:app --reload
```
Alternatively, you can test it with the Swagger UI by visiting:

```bash
http://127.0.0.1:8000/docs
```
Use the /predict/ endpoint with the following input:

```bash
{
  "content": "India wins the FIFA World Cup in 2026."
}
```
You will receive a JSON response like:

```bash
{
  "verdict": "Fake",
  "confidence": 91.32
}
```
If you are integrating this with Claude Desktop or an MCP server, you can define a wrapper that sends a POST request to the API. Once connected, you can call:

```bash
check_news("India wins the FIFA World Cup in 2026.")
```
