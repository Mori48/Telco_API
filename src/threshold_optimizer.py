from sklearn.base import BaseEstimator
from  sklearn.model_selection import StratifiedKFold

import logging
from sklearn.metrics import precision_recall_curve, f1_score
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import numpy as np

class ThresholdOptimizer(BaseEstimator):
    def __init__(self, base_model, metric="f1"):
        self.base_model = base_model
        self.metric = metric
        self.best_threshold_ = 0.5

    def fit(self, X, y):

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        oof_proba = np.zeros(len(y))

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            self.base_model.fit(X_tr, y_tr)


            oof_proba[val_idx] = self.base_model.predict_proba(X_val)[:, 1]

        thresholds = np.linspace(0.1, 0.9, 81)
        scores = []
        for t in thresholds:
            y_pred = (oof_proba >= t).astype(int)
            scores.append(f1_score(y, y_pred))


        self.best_threshold_ = thresholds[np.argmax(scores)]
        best_f1 = max(scores)

        self.base_model.fit(X, y)

    

        logger.info(f"Оптимальный порог = {self.best_threshold_:.3f}")
        logger.info(f"F1 = {best_f1:.3f}")

        return self

    def predict(self, X):
        y_proba = self.base_model.predict_proba(X)[:, 1]
        return (y_proba >= self.best_threshold_).astype(int)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)