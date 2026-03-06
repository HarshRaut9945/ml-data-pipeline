#Handle Missing Value
# outlier tratment
# Handle Imbalance dataset
# Convert categorucal coluns into numerical columns  

import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustmeException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


@dataclass
class DataTransfromartionConfigs:
      preprocess_obj_file_patrh = os.path.join("artifacts/data_transformation", "preprcessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransfromartionConfigs()


def  get_data_transformation_obj(self):
     try:
           logging.info(" Data Transformation Started")

           numerical_features = ['age', 'workclass',  'education_num', 'marital_status',
            'occupation', 'relationship', 'race', 'sex', 'capital_gain',
            'capital_loss', 'hours_per_week', 'native_country']
           
           num_pipeline=Pipeline(
                steps=[
                     ("imputer",SimpleImputer(strategy='median')),
                     ("scaler",StandardScaler())
                ]
           )
           preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features)
            ])
             
           return preprocessor

     except Exception as e:
            raise CustmeException(e, sys)