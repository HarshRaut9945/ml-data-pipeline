# Handle Missing Value
# Outlier Treatment
# Handle Imbalance Dataset
# Convert Categorical Columns into Numerical Columns

import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Configuration Class
@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join(
        "artifacts", "data_transformation", "preprocessor.pkl"
    )


# Main Transformation Class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    # Create preprocessing pipeline
    def get_data_transformation_obj(self):

        try:
            logging.info("Data Transformation Started")

            numerical_features = [
                'age',
                'education_num',
                'capital_gain',
                'capital_loss',
                'hours_per_week'
            ]

            categorical_features = [
                'workclass',
                'marital_status',
                'occupation',
                'relationship',
                'race',
                'sex',
                'native_country'
            ]

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            # Column Transformer
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features)
            ])

            logging.info("Preprocessing Pipeline Created")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    # Outlier Handling using IQR
    def remove_outliers_IQR(self, col, df):

        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            IQR = Q3 - Q1

            upper_limit = Q3 + 1.5 * IQR
            lower_limit = Q1 - 1.5 * IQR

            df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
            df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])

            return df

        except Exception as e:
            logging.info("Outlier handling failed")
            raise CustomException(e, sys)


    # Apply Transformation
    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info("Reading Train and Test Data")

            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            numerical_features = [
                'age',
                'education_num',
                'capital_gain',
                'capital_loss',
                'hours_per_week'
            ]

            # Handle Outliers for Train Data
            for col in numerical_features:
                train_data = self.remove_outliers_IQR(col, train_data)

            logging.info("Outliers handled in train data")

            # Handle Outliers for Test Data
            for col in numerical_features:
                test_data = self.remove_outliers_IQR(col, test_data)

            logging.info("Outliers handled in test data")


            preprocessing_obj = self.get_data_transformation_obj()

            target_column = "income"
            drop_columns = [target_column]

            logging.info("Splitting Input and Target Features")

            input_feature_train = train_data.drop(columns=drop_columns)
            target_feature_train = train_data[target_column]

            input_feature_test = test_data.drop(columns=drop_columns)
            target_feature_test = test_data[target_column]


            logging.info("Applying Preprocessing Object")

            input_train_arr = preprocessing_obj.fit_transform(input_feature_train)
            input_test_arr = preprocessing_obj.transform(input_feature_test)


            # Combine features and target
            train_arr = np.c_[input_train_arr, np.array(target_feature_train)]
            test_arr = np.c_[input_test_arr, np.array(target_feature_test)]


            logging.info("Saving Preprocessing Object")

            save_object(
                file_path=self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocessing_obj
            )


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)