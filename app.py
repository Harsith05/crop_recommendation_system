from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np

# Load trained objects
model = pickle.load(open("crop_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

app = FastAPI()

# Templates & Static
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class CropInput(BaseModel):
    nitrogen: int
    phosphorus: int
    potassium: int
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# ✅ HOME PAGE
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# ✅ PREDICTION API
@app.post("/predict")
def predict_crop(data: CropInput):

    input_data = np.array([[ 
        data.nitrogen,
        data.phosphorus,
        data.potassium,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]])

    scaled_data = scaler.transform(input_data)
    encoded_prediction = model.predict(scaled_data)

    crop_name = label_encoder.inverse_transform(encoded_prediction)[0]

    return {
        "recommended_crop": crop_name
    }
