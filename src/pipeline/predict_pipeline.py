import sys 
import pandas as pd 
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifact', 'model.pkl')
            preprocessor_path = os.path.join('artifact', 'preprocessor.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys.exc_info())

class CustomData:
    def __init__(self, 
                 List_Year: int, 
                 Assessed_Value: float, 
                 Sales_Ratio: float, 
                 Property_Type: str, 
                 Residential_Type: str, 
                 Town: str):
        self.List_Year = List_Year
        self.Assessed_Value = Assessed_Value
        self.Sales_Ratio = Sales_Ratio
        self.Property_Type = Property_Type
        self.Residential_Type = Residential_Type
        self.Town = Town

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'List Year': [self.List_Year],
                'Assessed Value': [self.Assessed_Value],
                'Sales Ratio': [self.Sales_Ratio],
                'Property Type': [self.Property_Type],
                'Residential Type': [self.Residential_Type],
                'Town': [self.Town]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys.exc_info())