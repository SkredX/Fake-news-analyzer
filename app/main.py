from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict_news

app = FastAPI(title="Fake News Detection API")

class NewsItem(BaseModel):
    content: str

@app.post("/predict/")
def get_prediction(item: NewsItem):
    return predict_news(item.content)
