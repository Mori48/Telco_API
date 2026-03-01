from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import io
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from src.predict import load_model, predict_churn, make_prediction
import uvicorn

import logging
from typing import List
app = FastAPI(title="Telco Churn Prediction API")

logger = logging.getLogger(__name__)

try:
    model = load_model()
    logger.info("Модель загружена успешно")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    pipeline = None

class CustomerData(BaseModel):
    # 1. Основная информация
    gender: str                       # Пол: "Male", "Female"
    SeniorCitizen: int                # Пенсионер: 0, 1
    Partner: str                      # Партнер: "Yes", "No"
    Dependents: str                   # Иждивенцы: "Yes", "No"
    
    # 2. Информация о услугах
    tenure: int                       # Сколько месяцев с нами
    PhoneService: str                 # Телефонная служба: "Yes", "No"
    MultipleLines: str                # Многоканальная связь: "Yes", "No", "No phone service"
    InternetService: str              # Интернет-сервис: "DSL", "Fiber optic", "No"
    OnlineSecurity: str               # Онлайн-безопасность: "Yes", "No", "No internet service"
    OnlineBackup: str                 # Онлайн-резервное копирование: "Yes", "No", "No internet service"
    DeviceProtection: str             # Защита устройства: "Yes", "No", "No internet service"
    TechSupport: str                  # Техническая поддержка: "Yes", "No", "No internet service"
    StreamingTV: str                  # Потоковое ТВ: "Yes", "No", "No internet service"
    StreamingMovies: str              # Потоковые фильмы: "Yes", "No", "No internet service"
    
    # 3. Информация о счетах
    Contract: str                     # Контракт: "Month-to-month", "One year", "Two year"
    PaperlessBilling: str             # Безбумажный биллинг: "Yes", "No"
    PaymentMethod: str                # Метод оплаты: "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    MonthlyCharges: float             # Ежемесячные charges
    TotalCharges: float               # Общие charges

@app.get("/")
def root():
    return {"status": "ok", "service": "Telco Churn Prediction API"}

@app.get("/app")
def home():
    return {"message": "Telco Churn Prediction API is working"}

@app.post("/predict/single")
async def predict_single(customer:CustomerData):
    try:
        new_data = pd.DataFrame([customer.model_dump()])
        predictions, probabilities = predict_churn(new_data, model)
        return {
            "churn_prediction": bool(predictions[0]),
            "churn_probability": float(probabilities[0][1]),
            "confidence": "high" if probabilities[0][1] > 0.7 else "medium"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/predict/batch")
async def predict_batch(customers: List[CustomerData]):
    try:
        # customers_dict = [customer for customer in customers]
        new_data = pd.DataFrame(customers)

        predictions, probabilities = predict_churn(new_data, model)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "customer_id": i,
                "churn_prediction": bool(pred),
                "churn_probability": float(prob[1]),
                "confidence": "high" if probabilities[0][1] > 0.7 else "medium"
            })
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/predict/file")
async def predict_file (file: UploadFile = File(...)):

    try:

        contents = await file.read()
        new_data = pd.read_csv(io.BytesIO(contents))


        make_prediction()
        predictions, probabilities = predict_churn(new_data, model)
    
        result_df = new_data.copy()
        result_df['Churn_Prediction'] = predictions
        result_df['Churn_Probability'] = probabilities[:, 1]
        
        results = result_df.to_dict(orient='records')
        
        return {
            "filename": file.filename,
            "predictions": results,
            "total_records": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)