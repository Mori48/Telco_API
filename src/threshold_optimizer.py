from sklearn.base import BaseEstimator, ClassifierMixin

import logging
from sklearn.metrics import precision_recall_curve
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import numpy as np

class ThresholdOptimizer(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, metric="recall"):
        self.base_model = base_model
        self.metric = metric
        self.best_threshold_ = 0.5

    def fit(self, X, y):
        self.base_model.fit(X, y)
        y_proba = self.base_model.predict_proba(X)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        self.best_threshold_ = thresholds[np.nanargmax(f1_scores)]

        logger.info(f"Оптимальный порог = {self.best_threshold_:.3f}")

        return self

    def predict(self, X):
        y_proba = self.base_model.predict_proba(X)[:, 1]
        return (y_proba >= self.best_threshold_).astype(int)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)