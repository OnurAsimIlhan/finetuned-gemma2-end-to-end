from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os
import json
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

import certifi
ca=certifi.where()

import sys
import pandas as pd
import numpy as np
import pymongo
from src.bluearf.exception.exception import NetworkSecurityException
from src.bluearf.logging.logger import logging

class DataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def excel_to_json_converter(self,file_path):
        try:
            data=pd.read_excel(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def insert_data_mongodb(self, records, database, collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
    
        except Exception as e:
            raise NetworkSecurityException(e,sys)

if __name__=='__main__':
    FILE_PATH="data/Bluearf Machine Learning Coding Task Data.xlsx"
    DATABASE="BLUEARF"
    Collection="ACTIVITY"
    networkobj=DataExtract()
    records=networkobj.excel_to_json_converter(file_path=FILE_PATH)
    print(records)
    no_of_records=networkobj.insert_data_mongodb(records,DATABASE,Collection)
    print(no_of_records)