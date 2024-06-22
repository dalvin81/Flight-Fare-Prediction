import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, load_preprocessor


# mapping html input to trained model
class CustomData:
    def __init__(self,
                 airline:str,
                 source_city:str,
                 destination_city:str,
                 departure_time:str,
                 arrival_time:str,
                 Class:str,
                 stops:str,
                 days_left:int,
                 duration:float
                 ):
        self.airline = airline
        self.source_city = source_city
        self.destination_city = destination_city
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        self.Class = Class
        self.stops = stops
        self.days_left = days_left
        self.duration = duration
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "airline": [self.airline],
                "source_city": [self.source_city],
                "destination_city": [self.destination_city],
                "departure_time": [self.departure_time],
                "arrival_time": [self.arrival_time],
                "class": [self.Class],
                "stops": [self.stops],
                "days_left": [self.days_left],
                "duration": [self.duration]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
    

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor, labelEncoders  =load_preprocessor(file_path=preprocessor_path)
            for col in labelEncoders:
                le = labelEncoders[col]
                features[col] = le.transform(features[col])
        
            data_scaled = preprocessor.transform(features)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)