import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessing_obj_file_path= os.path.join('artifacts','Preprocessing.pkl')

class Datatransformation:
    def __init__(self):
        self.Datatransformation_config=DataTransformationconfig()


    def get_data_transformation_object(self):
        try:
            numerical_features=['writing score','reading score']
            categorical_features=[
                "gender",
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course',
            ]
            num_pipeline=Pipeline(
                steps=[
                ("Imputer", SimpleImputer(strategy='median')),
                ("StandardScaler",StandardScaler()),
                   

                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ("Imputer",SimpleImputer(strategy='most_frequent')),
                ("One_Hot_Encoder",OneHotEncoder()),
                ("Standard_Scaler",StandardScaler(with_mean=False))

                ]
            )

            
            logging.info(f"Categorical  Columns: {categorical_features}")
            logging.info(f"Numerical columns : {numerical_features}")

            


            preprocessor=ColumnTransformer(


                [
                ("num_pipeline",num_pipeline,numerical_features),
                ("cat_pipeline",cat_pipeline,categorical_features)

                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)

        

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)
            

            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessing objects')

            preprocessing_object=self.get_data_transformation_object()

            target_column_name= "math score"
            numerical_columns=["writing  score","reading score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df= train_df[target_column_name]

            Input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on train and test Dataframes")

            input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.transform(Input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]

            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('Saved preprocessing object')

            save_object(
                file_path=self.Datatransformation_config.preprocessing_obj_file_path,
                obj=preprocessing_object
            )
            return(
                train_arr,
                test_arr,
                self.Datatransformation_config.preprocessing_obj_file_path,
            )




        except Exception as e:
            raise CustomException(e,sys)



        


            

            



