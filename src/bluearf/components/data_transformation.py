import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from src.bluearf.constants.training_pipeline import TARGET_COLUMN
from src.bluearf.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from src.bluearf.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from src.bluearf.entity.config_entity import DataTransformationConfig
from src.bluearf.exception.exception import NetworkSecurityException 
from src.bluearf.logging.logger import logging
from src.bluearf.utils.main_utils.utils import save_numpy_array_data,save_object, extract_emission_factors

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            train_df = train_df.drop(columns=['ACTIVITY_ID', 'ID', 'DATA_VERSIONING', 'YEAR_RELEASED', 'LCA_ACTIVITY'])
            test_df = test_df.drop(columns=['ACTIVITY_ID', 'ID', 'DATA_VERSIONING', 'YEAR_RELEASED', 'LCA_ACTIVITY'])

            train_df['EXTRACTED_FACTORS'] = extract_emission_factors(train_df['EMISSION_FACTORS'])
            test_df['EXTRACTED_FACTORS'] = extract_emission_factors(test_df['EMISSION_FACTORS'])

            factors_df = pd.DataFrame(train_df['EXTRACTED_FACTORS'].tolist())
            factors_df_test = pd.DataFrame(test_df['EXTRACTED_FACTORS'].tolist())

            result_data = pd.concat([train_df, factors_df], axis=1)
            train_result_data = result_data.drop(columns=['EXTRACTED_FACTORS', 'EMISSION_FACTORS', 'CO2E_CALCULATION_METHOD'])

            result_data = pd.concat([test_df, factors_df_test], axis=1)
            test_result_data = result_data.drop(columns=['EXTRACTED_FACTORS', 'EMISSION_FACTORS', 'CO2E_CALCULATION_METHOD'])

            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_result_data, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_result_data,)

            #preparing artifacts

            data_transformation_artifact=DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact


            
        except Exception as e:
            raise NetworkSecurityException(e,sys)