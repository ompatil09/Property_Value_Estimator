from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import pickle


data=pd.read_csv('train.csv')
# for column in data.columns:
    # print(data[column].value_counts())
    # print("*"*20)
data.isna().sum()
data.drop(columns=['lot_size','lot_size_units','size_units'],inplace=True)
data['price_per_sqft'] = data['price'] * 100000 / data['size']
data['price_per_sqft']
data.drop(columns=['price_per_sqft'],inplace=True)
data.to_csv("final_dataset.csv")
X=data.drop(columns=['price'])
y=data['price']    
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)
# print(X_train.shape)
# print(y_train.shape)
ridge = Ridge()
pipe = make_pipeline(StandardScaler(), ridge)
pipe.fit(X_train,y_train)
y_pred_ridge = pipe.predict(X_test)
r2=r2_score(y_test,y_pred_ridge)
print(r2)
pickle.dump(pipe, open('RidgeModel.pkl','wb'))
