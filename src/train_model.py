
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline  
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



def Log_Reg_model (X_train, y_train, preprocessor,best_params):
        
    logger.info('Обучение XGBoost')
    pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(**best_params))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline



def train_and_save_pipeline(data_path, params_path, model_save_path):
    df = load_raw_file(data_path)
    X_train, X_test, y_train, y_test, preprocessor = preprocessing(df)
    best_params = load_best_params(params_path)
    
    pipeline = Log_Reg_model(X_train, y_train, preprocessor, best_params)
    y_pred = pipeline.predict(X_test)
    logger.info(classification_report(y_test,y_pred))

    joblib.dump(pipeline,model_save_path,compress=6 )

    logger.info("Финальная модель обучена успешно")

    return pipeline


if __name__ == "__main__":
    pipeline = train_and_save_pipeline(
        data_path='data\WA_Fn-UseC_-Telco-Customer-Churn.csv',
        params_path='config\model.json', 
        model_save_path='model\Log_Reg_model.joblib'
    )

    logger.info("Обучение завершено")
