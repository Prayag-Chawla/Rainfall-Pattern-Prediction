
# Raunfall Patern Predciction
Rainfall Pattern detection is one of the most important real-life problems, especially in the agricultural sector. Weather pattern forecasting is essential to understand and analyze factors like the right crop that needs to be grown, the best time to harvest, precautions to undertake on the field, etc. Weather plays a crucial role in the whole agricultural process. It is not that easy to predict the course of weather accurately and here is where Machine Learning and AI come to play. Highly trained models tuned with the right hyper-parameters can yield accurate results in forecasting. The project covers the detection of rainfall patterns using some machine learning models.


## Description

In this project, we are taking the rainfall data of Tamil Nadu, and based on the rainfall pattern during a three-month window, we try predicting the succesive month's rainfall pattern. Since this is a very basic project, a lot of importance wasn't given to the tuning of the hyperparameters. We have used three ML models - Random Forest, Lasso and Linear Regression models.

##NOTE - 
It is to be noted that I was working on a high end rinfall prediction project with much more deepe analysis and using multiple models at the same time, then comparing it all at the same time via various methds of visualization.
I will uploat the notebook for the same with the dataset.
I ran into a few problems while I was running it locally in my vscode, but the overll semantics of the project remains.

## Acknowledgements

The data was taken from : https://www.kaggle.com/rajanand/rainfall-in-india?select=rainfall+in+india+1901-2015.csv
## Usage and Installation

```

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

rf = RandomForestRegressor(n_estimators = 100, max_depth=10, n_jobs=1)
rf.fit(X_train, y_train)

linear_regressor = LinearRegression()  
linear_regressor.fit(X_train, y_train)  

import statsmodels.api as sm

import statsmodels.formula.api as smf










The australia one - 

from sklearn.preprocessing import LabelEncoder

from sklearn.utils import resample

import warnings
from sklearn import preprocessing
r_scaler = preprocessing.MinMaxScaler()
r_scaler.fit(MiceImputed)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as rf

from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, roc_curve, classification_report

from runpy import run_module
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

import catboost as cb

import xgboost as xgb

```
After this, we checked for accuracy in each and every model at the same time.





## Used By

The project is used by a lot of weather and forecasting companies to analyse the environment. moreover, its used by news achannel and scientists to deeply analyse all the disturbances in our environment.Therefore, it holds a lot of credibility with itself.


## References
-  https://github.com/vgaurav3011/Rainfall-Prediction/blob/master/Exploration_Rainfall_Data.ipynb

 - https://thecleverprogrammer.com/2020/09/11/rainfall-prediction-with-machine-learning/

I will give you a reference of all the models used in machine learning in case you are wondering

https://www.javatpoint.com/machine-learning-algorithms