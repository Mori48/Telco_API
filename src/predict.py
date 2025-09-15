import joblib
import logging
import os
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.threshold_optimizer import ThresholdOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model():
    try:
        logger.info('Загрузка модели')
        model_path = os.path.join('model', 'XGB_model.joblib')
        model = joblib.load(model_path)
        logger.info('Модель загружена')
        return model 
    except FileNotFoundError:
        logger.error('Модель не найдена')
        return None

def predict_churn(new_data,model):
    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)
    return predictions, probabilities

def make_prediction(new_data_folder = 'new_data',output_folder = 'predictions'):
    model = load_model()
    for filename in os.listdir(new_data_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(new_data_folder, filename)
            logger.info('Чтение файла')
        try:
            new_data = pd.read_csv(file_path)
            predictions, probabilities = predict_churn(new_data, model)

            result_df = new_data.copy()
            result_df['Churn_Prediction'] = predictions
            result_df['Churn_Probability_0'] = probabilities[:, 0]  
            result_df['Churn_Probability_1'] = probabilities[:, 1] 

            output_filename = f"predicted_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            result_df.to_csv(output_path, index=False)
            
            logger.info(f"Предсказания сохранены в: {output_path}")
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {filename}: {str(e)}")


if __name__ == "__main__":
    make_prediction()
