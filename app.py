from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from scipy import stats
from math import log
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import re
from os.path import dirname, join, realpath





class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return  mod.predict(items)[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return mod.predict(items)

if __name__ == "__main__":
    class modelRegressor():
        def __init__(self):
          self.pipe = Pipeline([('scaler', StandardScaler()),
                          ('Regressor', Ridge(alpha=2))])
          self.base_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque',
            'max_torque_rpm', 'name_Audi', 'name_BMW',
           'name_Chevrolet', 'name_Daewoo', 'name_Datsun', 'name_Fiat',
           'name_Force', 'name_Ford', 'name_Honda', 'name_Hyundai', 'name_Isuzu',
           'name_Jaguar', 'name_Jeep', 'name_Kia', 'name_Land', 'name_Lexus',
           'name_MG', 'name_Mahindra', 'name_Maruti', 'name_Mercedes-Benz',
           'name_Mitsubishi', 'name_Nissan', 'name_Peugeot', 'name_Renault',
           'name_Skoda', 'name_Tata', 'name_Toyota', 'name_Volkswagen',
           'name_Volvo', 'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol',
           'seller_type_Individual', 'seller_type_Trustmark Dealer',
           'transmission_Manual', 'owner_Fourth & Above Owner',
           'owner_Second Owner', 'owner_Test Drive Car', 'owner_Third Owner',
           'seats_4', 'seats_5', 'seats_6', 'seats_7', 'seats_8', 'seats_9',
           'seats_10', 'seats_14']
          self.pro_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque',
           'max_torque_rpm', 'name_Audi', 'name_BMW', 'name_Chevrolet',
           'name_Daewoo', 'name_Datsun', 'name_Fiat', 'name_Force', 'name_Ford',
           'name_Honda', 'name_Hyundai', 'name_Isuzu', 'name_Jaguar', 'name_Jeep',
           'name_Kia', 'name_Land', 'name_Lexus', 'name_MG', 'name_Mahindra',
           'name_Maruti', 'name_Mercedes-Benz', 'name_Mitsubishi', 'name_Nissan',
           'name_Peugeot', 'name_Renault', 'name_Skoda', 'name_Tata',
           'name_Toyota', 'name_Volkswagen', 'name_Volvo', 'fuel_Diesel',
           'fuel_LPG', 'fuel_Petrol', 'seller_type_Individual',
           'seller_type_Trustmark Dealer', 'transmission_Manual',
           'owner_Fourth & Above Owner', 'owner_Second Owner',
           'owner_Test Drive Car', 'owner_Third Owner', 'seats_4', 'seats_5',
           'seats_6', 'seats_7', 'seats_8', 'seats_9', 'seats_10', 'seats_14',
           'new1', 'new111', 'new1.1', 'new2', 'new3', 'new3.1', 'new3.2', 'new4',
           'new5', 'new6', 'new10', 'new11', 'new12', 'new13', 'premium', 'cheap',
           'more_owner', 'brand_new', 'new3.3', 'new3.4', 'new112']
    
        def fit(self,Xs,ys):
          self.pipe.fit(Xs,ys)
    
    
        def data_transform(self,df):
          df['name'] = df['name'].apply(lambda x: x.split()[0])
    
          enc = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True, dtype = int)
          enc = pd.get_dummies(enc, columns=['seats'], drop_first=True, dtype = int)
          enc = enc.reindex(columns = self.base_cols, fill_value=0)
          return enc
        
        def new_features(self,df):
            df['new1'] = df['engine'] * df['max_power']
            df['new111'] = df['engine'] * df['km_driven']
            
            df['new1.1'] = df['engine'] / df['max_power']
            df['new2' ]= (df['mileage'] * df['mileage'])
            df['new3' ]= (df['new1'] * (df['year']-2017))
            df['new3.1' ]= (df['new111'] * (df['year']-2017))
            df['new3.2' ]= (df['new1.1'] * (df['year']-2017))
            for i in range(4,7):
              df[f'new{i}' ]= (df['mileage'] *(df[f'seats_{i}']))
              df['new10'] = df['torque'] * df['engine']
              df['new11'] = (df['year']-2017) ** 2
              df['new12'] = (df['mileage']) ** 2
              df['new13'] = (df['fuel_Diesel']) * df['engine']
              df['new13'] = (df['fuel_Petrol']) * df['engine']
              
            ## часть 2
            df['premium'] = (df['name_Audi']) + (df['name_BMW']) + (df['name_Lexus']) + (df['name_Jaguar']) + (df['name_Mercedes-Benz'])
            df['cheap'] = (df['name_Maruti']) + (df['name_Mahindra']) + (df['name_Force'])
            
            
            df['more_owner'] = (df['owner_Third Owner']) + (df['owner_Fourth & Above Owner'])
            df['brand_new'] = (1 - df['owner_Fourth & Above Owner'] - df['owner_Second Owner'] - df['owner_Third Owner'] - df['owner_Test Drive Car']) * (1 - df['seller_type_Trustmark Dealer'] + df['seller_type_Individual'])
            df['new3.3' ]= (df['premium'] * (df['year']-2017))
            df['new3.4' ]= (df['cheap'] * (df['year']-2017))
            
            
            #Часть 3
            #Логарифмируем признаки с странным распределением
            df['torque'] = df['torque'].apply(lambda x: log(x))
            df['max_torque_rpm'] = df['max_torque_rpm'].apply(lambda x: log(x))
            df['new112'] = df['torque'] / df['max_torque_rpm']
            return df
    
        def predict(self,X_pred):
          t = self.new_features(self.data_transform(X_pred).reindex(columns = self.base_cols, fill_value=0)).reindex(columns = self.pro_cols, fill_value=0)
          print(t.columns)
          y_pred = self.pipe.predict(t)
          return y_pred
    with open("pipeline.pkl", 'rb') as f:
        mod = pickle.load(f)
    app = FastAPI()