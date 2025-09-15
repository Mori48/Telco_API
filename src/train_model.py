
import joblib
import json
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline  
from imblearn.over_sampling import SMOTE
import logging
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from src.threshold_optimizer import ThresholdOptimizer
import json


from src.preprocessing import load_raw_file, preprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_best_params(params_path='config\model.json'): 
    try:
        logger.info(f"Загрузка параметров из {params_path}")
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        logger.info(f"Загружены параметры: {best_params}")
        return best_params
    except FileNotFoundError:
        logger.error(f"Файл {params_path} не найден")
        raise
    except json.JSONDecodeError:
        logger.error(f"Ошибка чтения JSON файла {params_path}")
        raise



def xgb_model (X_train, y_train, preprocessor,best_params):
    xgb_params = best_params.get('params')
        
    logger.info('Обучение XGBoost')


    xgb = XGBClassifier(**xgb_params, random_state=14)
    
    pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=14)),
    ('classifier', ThresholdOptimizer(xgb))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline



def train_and_save_pipeline(data_path, params_path, model_save_path):
    df = load_raw_file(data_path)
    X_train, X_test, y_train, y_test, preprocessor = preprocessing(df)
    best_params = load_best_params(params_path)
    
    xgb_pipline = xgb_model(X_train, y_train, preprocessor, best_params)
    y_pred = xgb_pipline.predict(X_test)
    logger.info(classification_report(y_test,y_pred))

    joblib.dump(xgb_pipline,model_save_path,compress=9 )

    logger.info("Финальная модель обучена успешно")

    return xgb_pipline


if __name__ == "__main__":
    pipeline = train_and_save_pipeline(
        data_path='data\WA_Fn-UseC_-Telco-Customer-Churn.csv',
        params_path='config\model.json', 
        model_save_path='model\XGB_model.joblib'
    )

    logger.info("Обучение завершено")
