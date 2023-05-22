import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        directory = os.path.dirname(file_path)

        os.makedirs(directory, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, X_test, y_train, y_test, models):
    try:
        report={}
        for i in range(len(list(models.keys()))):
            model=models[list(models.keys())[i]]
            model.fit(X_train, y_train)
            predict=model.predict(X_test)
            test_score=r2_score(y_test, predict)
            report[list(models.keys())[i]]=test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)