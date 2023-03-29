import os
import sys
from dataclasses import dataclass 

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class modeltrainerconfig:
    Trained_model_file_path=os.path.join('artifacts','Model.pkl')

class modelTrainer:
    def __init__(self):
        self.model_trainer_config=modeltrainerconfig()


    def initiate_modeltrainer(self,train_array,test_array):
        try:

            logging.info('Spliting data')

            x_train,y_train,x_test,y_test=(

                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models= {
            
                
                "Linear Regression": LinearRegression(),
                "Gradient Boost": GradientBoostingRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Adaboost Regressor": AdaBoostRegressor(),
                "cat Boost": CatBoostRegressor(),
                "Random forest ": RandomForestRegressor(),
                "xg Boost": XGBRegressor(),
                "KNeighbors": KNeighborsRegressor(),


            }

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))

            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]



            best_model= models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best Model Found")
            logging.info(f"Best model found on both train and test dataset")

            save_object(
                file_path=self.model_trainer_config.Trained_model_file_path,
                obj= best_model
            )
            predicts= best_model.predict(x_test)
            r2_square=r2_score(y_test,predicts)
            return r2_square
        except Exception as e:
         
         raise CustomException(e,sys)
            
