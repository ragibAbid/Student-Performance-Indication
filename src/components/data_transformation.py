import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformer:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.num_features = ["reading_score","writing_score"]
        self.cat_features = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

    def get_transformer_obj(self):
        '''
        This function will create the data transformation pipeline.
        '''
        try:
            num_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info(f'Numerical Transformer: {num_pipeline}')

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical transformer: {cat_pipeline}")

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, self.num_features),
                    ('cat_pipeline', cat_pipeline, self.cat_features)
                ]
            )
            logging.info(f'Preprocessor: {preprocessor}')
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Data reading complete")

            preprocessor_obj = self.get_transformer_obj()
            logging.info('Got preprocessor object')

            target_column='math_score'
            input_feature_train_df=train_df.drop(columns=[target_column, 'Unnamed: 0'],axis=1)
            target_feature_train_df=train_df[target_column]
            logging.info(f"X_train Columns: {input_feature_train_df.columns}")

            input_feature_test_df=test_df.drop(columns=[target_column, 'Unnamed: 0'],axis=1)
            target_feature_test_df=test_df[target_column]
            logging.info(f'X_train=, X_test, y_train, y_test created')

            transformed_X_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            transformed_X_test_arr = preprocessor_obj.transform(input_feature_test_df)
            logging.info("Input features preprocessed.")

            train_arr=np.c_[transformed_X_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_X_test_arr, np.array(target_feature_test_df)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_path, obj=preprocessor_obj)
            logging.info("Preprocessor obj saved.")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_path
        except Exception as e:
            raise CustomException(e, sys)