from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# -----------------------------
# Load Model & Label Encoder
# -----------------------------

with open("crop_recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -----------------------------
# Create FastAPI App
# -----------------------------

app = FastAPI()

# -----------------------------
# Request Body Structure
# -----------------------------

class CropInput(BaseModel):
    Nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# -----------------------------
# Home Route
# -----------------------------

@app.get("/")
def home():
    return {"message": "🌱 Crop Recommendation API is running successfully!"}

# -----------------------------
# Prediction Route
# -----------------------------

@app.post("/predict")
def predict_crop(data: CropInput):

    input_df = pd.DataFrame([[
        data.Nitrogen,
        data.phosphorus,
        data.potassium,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]],
    columns=['Nitrogen', 'phosphorus', 'potassium',
             'temperature', 'humidity', 'ph', 'rainfall'])

    prediction = model.predict(input_df)
    crop_name = le.inverse_transform(prediction)[0]

    return {
        "recommended_crop": crop_name
    }
