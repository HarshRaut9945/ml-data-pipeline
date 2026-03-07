import os
import sys
import numpy as np
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts", "model_trainer", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define Models
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic": LogisticRegression()
            }

            # Hyperparameters
            params = {

                "Random Forest": {
                    "class_weight": ["balanced"],
                    "n_estimators": [20, 50, 100],
                    "max_depth": [5, 8, 10],
                    "min_samples_split": [2, 5, 10]
                },

                "Decision Tree": {
                    "class_weight": ["balanced"],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [3, 4, 5, 6],
                    "min_samples_split": [2, 3, 4]
                },

                "Logistic": {
                    "class_weight": ["balanced"],
                    "penalty": ["l1", "l2"],
                    "C": [0.001, 0.01, 0.1, 1, 10],
                    "solver": ["liblinear"]
                }
            }

            model_report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # Best Model
            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(
                f"Best Model Found: {best_model_name} | Accuracy: {best_model_score}"
            )

            logging.info(
                f"Best model found: {best_model_name} with accuracy: {best_model_score}"
            )

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

        except Exception as e:
            raise CustomException(e, sys)