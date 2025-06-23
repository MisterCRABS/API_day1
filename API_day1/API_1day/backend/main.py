from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from torchvision import models, transforms
from PIL import Image
import torch
import io
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

app = FastAPI()

# Загрузка моделей при старте сервера
@app.on_event("startup")
async def load_models():
    # Инициализация модели для классификации изображений (ResNet18)
    global img_model, img_transform
    img_model = models.resnet18(pretrained=True)
    img_model.eval()
    
    # Преобразования для изображения
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Инициализация модели для классификации текста (DistilBERT)
    global text_model, text_tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    text_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    text_tokenizer = AutoTokenizer.from_pretrained(model_name)
    global text_classifier
    text_classifier = pipeline("text-classification", model=text_model, tokenizer=text_tokenizer)

class TextRequest(BaseModel):
    text: str

@app.post("/classify_image/")
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    # Чтение и преобразование изображения
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img_t = img_transform(img).unsqueeze(0)
    
    # Предсказание
    with torch.no_grad():
        outputs = img_model(img_t)
    
    # Получение результата
    _, predicted = torch.max(outputs, 1)
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    
    return {"class": classes[predicted[0]], "class_id": int(predicted[0])}

@app.post("/classify_text/")
async def classify_text(request: TextRequest):
    result = text_classifier(request.text)
    return {
        "label": result[0]['label'],
        "score": float(result[0]['score'])
    }

@app.get("/")
async def root():
    return {"message": "Добро пожаловать в API классификации!"}