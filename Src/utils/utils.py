import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger.logger import logging


def save_object(file_path: str, obj) -> None:
    """
    Save a Python object to a file using dill for complex serialization.

    Args:
        file_path (str): Path where the object will be saved.
        obj: The object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {str(e)}")
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Load a Python object from a dill file.

    Args:
        file_path (str): Path to the saved object.

    Returns:
        Object loaded from the file.
    """
    try:
        with open(file_path, "rb") as file_obj:
            logging.info(f"Loading object from {file_path}")
            return dill.load(file_obj)
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {str(e)}")
        raise CustomException(e, sys)


def evaluate_models(X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray, 
                    models: dict, params: dict) -> dict:
    """
    Evaluate multiple machine learning models using GridSearchCV and calculate their R^2 scores.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing target values.
        models (dict): Dictionary containing model names and instantiated model objects.
        params (dict): Dictionary containing model names and hyperparameter grids for GridSearchCV.

    Returns:
        dict: A dictionary containing model names and their corresponding R^2 test scores.
    """
    try:
        report = {}

        for model_name, model in models.items():
            try:
                logging.info(f"Evaluating model: {model_name}")
                param_grid = params.get(model_name, {})

                # Perform Grid Search with Cross-Validation
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)

                # Update model with the best parameters found by GridSearchCV
                best_model = gs.best_estimator_

                # Predict on both training and testing data
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                # Calculate R^2 scores
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                # Log the results
                logging.info(f"{model_name} | Train R^2: {train_r2:.4f} | Test R^2: {test_r2:.4f}")

                # Store the R^2 score for the test data in the report
                report[model_name] = test_r2

            except Exception as model_error:
                # If there's an error with a specific model, log it and continue with others
                logging.error(f"Error evaluating model {model_name}: {str(model_error)}")

        return report

    except Exception as e:
        logging.error(f"Error during model evaluation: {str(e)}")
        raise CustomException(e, sys)
