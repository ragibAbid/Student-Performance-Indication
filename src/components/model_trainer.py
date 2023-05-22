# Basic Import
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from dataclasses import dataclass
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_path= os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train,X_test,y_train,y_test=(
                train_arr[:,:-1],
                test_arr[:,:-1], 
                train_arr[:,-1], 
                test_arr[:,-1]
                )
            logging.info("train test split complete")
            
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
                }
            logging.info("models defined")

            model_report: dict = evaluate_models(
                                    X_train=X_train, 
                                    y_train=y_train, 
                                    X_test=X_test,
                                    y_test=y_test, 
                                    models=models)
            
            best_model_score= max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No models satisfactory")
            logging.info(f"Best model found: {best_model_name}; score{best_model_score}")
            
            save_object(file_path=self.model_trainer_config.trained_model_path, obj=best_model)
            logging.info('Best model saved!')

            pred=best_model.predict(X_test)
            r2=r2_score(y_test, pred)
            return r2
        except Exception as e:
            raise CustomException(e, sys)