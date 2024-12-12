import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {"criterion": ["squared_error", "friedman_mse"]},
                "Random Forest": {"n_estimators": [8, 16, 32, 64]},
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [8, 16],
                },
                "XGBRegressor": {"learning_rate": [0.1, 0.01], "n_estimators": [8, 16]},
                "CatBoost Regressor": {"depth": [6, 8], "iterations": [30, 50]},
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [8, 16],
                },
            }

            model_report = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            predicted = best_model.predict(X_test)
            return r2_score(y_test, predicted)

        except Exception as e:
            raise CustomException(e, sys)
