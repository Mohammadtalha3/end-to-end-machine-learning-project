import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass #It is better to use @dataclass when just declaring variable instead of constructor but better to use constructor when ned to make functions
class dataingestionconfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path= os.path.join('artifacts','data.csv')




class DataIngestion:
    def __init__(self):
        self.Ingestion_config= dataingestionconfig() #this will be storing the path of the upper defined paths 


    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            df=pd.read_csv("E:\\end to end machine learning\\Notebook\\Data\\StudentsPerformance.csv")#Reading Dataset
            logging.info('Reading the dataset as DataFrame')

            os.makedirs(os.path.dirname(self.Ingestion_config.train_data_path),exist_ok=True)#making directory for train data path
            df.to_csv(self.Ingestion_config.raw_data_path,index=False,header=True)#raw data to csv

            logging.info("Train Test Split Initiated")

            train_set,test_set= train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.Ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.Ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of the data s completed')

            return(
                self.Ingestion_config.train_data_path,
                self.Ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":

    obj= DataIngestion()
    obj.initiate_data_ingestion()


