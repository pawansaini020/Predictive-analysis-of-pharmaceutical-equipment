# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:26:41 2019

@author: Basant
"""

import numpy as np 
from sklearn import datasets, linear_model, metrics 
import pandas as pd
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import scipy

data=pd.read_csv('Concrete_Data.csv',skiprows=1)

data=np.array(data)
X=data[:,0:8]
Y=data[:,8]

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)

reg = linear_model.LinearRegression() 
  
# train the model using the training sets 
reg.fit(x_train, y_train) 

aa=reg.predict(x_test)
print('prediction',aa)
from sklearn.metrics import mean_squared_error
y_pred=aa
ans=mean_squared_error(y_test, y_pred)
print('mean square error',ans)
