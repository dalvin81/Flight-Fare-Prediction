import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_object

@dataclass
class DataTransformConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransform:
    def __init__(self):
        self.data_transform_config = DataTransformConfig()
        
    def label_encode_categorical_features(self, X, categorical_cols, fit=True, encoders=None):
        try:
            if fit:
                label_encoders = {}
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    label_encoders[col] = le
                logging.info("Categorical columns label encoding completed (fit mode)")
                return X, label_encoders
            else:
                for col in categorical_cols:
                    le = encoders[col]
                    X[col] = le.transform(X[col])
                logging.info("Categorical columns label encoding completed (transform mode)")
                return X

        except Exception as e:
            raise CustomException(e, sys)
        
    def get_data_transform_obj(self):
        try:
            numerical_cols = ['duration', 'days_left']
            categorical_cols = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time',
       'destination_city', 'class']
            
            num_pipeline = Pipeline(
                steps = [
                    ("scaler",StandardScaler())
                ]
            )
            
            
            logging.info("Numerical columns transformation pipeline created"),
            
            
            preprocessor = ColumnTransformer(
                    transformers=[("num_pipeline", num_pipeline, numerical_cols)]
                ,
                remainder='passthrough'
            )
            
            return preprocessor, categorical_cols
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading train and test data completed")
            
            train_df.drop(labels=train_df.columns[0], axis=1, inplace=True)
            test_df.drop(labels=test_df.columns[0], axis=1, inplace=True)
            train_df.drop('flight', axis=1, inplace=True)
            test_df.drop('flight', axis=1, inplace=True)
            
            target = "price"
            X_train = train_df.drop(target,axis=1)
            y_train = train_df[target]
            X_test = test_df.drop(target, axis=1)
            y_test = test_df[target]
            
            logging.info("Encoding and transforming training and test data")
            
            preprocessor, categorical_cols = self.get_data_transform_obj()
            X_train, label_encoders = self.label_encode_categorical_features(X_train, categorical_cols, fit=True)
            X_train_preprocessed = preprocessor.fit_transform(X_train)
            X_test = self.label_encode_categorical_features(X_test, categorical_cols, fit=False, encoders=label_encoders)
            X_test_preprocessed = preprocessor.transform(X_test)
            
            logging.info("Data transformation completed successfully for both training and test sets")
            
            train_arr = np.c_[
                X_train_preprocessed, np.array(y_train)
            ]
            
            test_arr = np.c_[
                X_test_preprocessed, np.array(y_test)
            ]
            
            save_object(
                file_path = self.data_transform_config.preprocessor_obj_file_path,
                obj = (preprocessor, label_encoders)
            )
            

            return (
                train_arr,
                test_arr,
                self.data_transform_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e,sys)
