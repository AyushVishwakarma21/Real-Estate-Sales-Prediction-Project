import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['List Year', 'Assessed Value', 'Sales Ratio']
            categorical_columns = ['Property Type', 'Residential Type']
            town_features = ['Town']

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Ensure dense output
            ])

            town_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Ensure dense output
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns),
                    ('town', town_pipeline, town_features)
                ],
                remainder='drop',  # Changed to drop extra columns
                sparse_threshold=0  # Force dense matrix output
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys.exc_info())
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            columns_to_drop = [
                'Serial Number',
                'Date Recorded',
                'Address',
                'Non Use Code',
                'Assessor Remarks',
                'OPM remarks',
                'Location'
            ]
            train_df = train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns])
            test_df = test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns])

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Sale Amount"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets")
            
            # Transform input features (returns dense arrays now)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Reshape target variables
            target_train_arr = target_feature_train_df.values.reshape(-1, 1)
            target_test_arr = target_feature_test_df.values.reshape(-1, 1)

            # Combine features and targets
            train_arr = np.hstack((input_feature_train_arr, target_train_arr))
            test_arr = np.hstack((input_feature_test_arr, target_test_arr))

            logging.info(f"Saved preprocessing object at {self.data_transformation_config.preprocessor_obj_file_path}")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys.exc_info())