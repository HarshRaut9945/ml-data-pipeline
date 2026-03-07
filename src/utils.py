from src.logger import logging
from src.exception import CustomException
import os
import sys
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name in models:

            model = models[model_name]
            para = params[model_name]

            gs = GridSearchCV(model, para, cv=5)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)

            test_model_accuracy = accuracy_score(y_test, y_pred)

            report[model_name] = test_model_accuracy

        return report

    except Exception as e:
        raise CustomException(e, sys)