#!/usr/bin/env python
# coding: utf-8

# In[1]:



import joblib
from sklearn.model_selection import train_test_split
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

import numpy as np

import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import sys
# In[3]:


from sklearn import svm
from sklearn.model_selection import GridSearchCV


from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[4]:


train=pd.read_csv(sys.argv[1])



# In[42]:


y1=train['energy_per_atom']
y2=train['packing fraction']


# In[43]:



# In[7]:


train


# In[6]:


X_train=train.iloc[:,14:]
X_train

from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train = scaler.transform(X_train)

# In[ ]:


from sklearn.model_selection import GridSearchCV, RepeatedKFold
param_grid = {"C": [i for i in range(1, 600, 5)],'gamma':[10**(-a) for a in range(-1,10)],'epsilon':[5*10**(-a) for a in range(-1,6)],'kernel':['rbf']}
# Define svr here
...
svr = svm.SVR()
# Specify cross-validation generator, in this case (10 x 5CV)
cv = RepeatedKFold(n_splits=5, n_repeats=10)
clf = GridSearchCV(estimator=svr, param_grid=param_grid, scoring="neg_mean_squared_error", cv=cv,n_jobs=-1)

# Continue as usual
clf.fit(X_train, y1)



print('Energy_best_score',clf.best_score_)
print('Energy_best_parmeters',clf.best_params_)
a=pd.DataFrame(clf.cv_results_)[['param_gamma','param_C','param_kernel','mean_test_score']]
a.to_csv(sys.argv[2])



# In[ ]:


from sklearn.model_selection import GridSearchCV, RepeatedKFold
param_grid = {"C": [i for i in range(1, 600, 5)],'gamma':[10**(-a) for a in range(-1,10)],'epsilon':[5*10**(-a) for a in range(-1,6)],'kernel':['rbf']}
# Define svr here
...
svr = svm.SVR()
# Specify cross-validation generator, in this case (10 x 5CV)
cv = RepeatedKFold(n_splits=5, n_repeats=10)
clf = GridSearchCV(estimator=svr, param_grid=param_grid, scoring="neg_mean_squared_error", cv=cv,n_jobs=-1)

# Continue as usual
clf.fit(X_train, y2)



print('Packing_fraction_best_score',clf.best_score_)
print('Packing_fraction_best_parameers',clf.best_params_)
a=pd.DataFrame(clf.cv_results_)[['param_gamma','param_C','param_kernel','mean_test_score']]
a.to_csv(sys.argv[3])


# In[ ]:


y3=train['formation_energy_per_atom']
from sklearn.model_selection import GridSearchCV, RepeatedKFold
param_grid = {"C": [i for i in range(1, 600, 5)],'gamma':[10**(-a) for a in range(-1,10)],'epsilon':[5*10**(-a) for a in range(-1,6)],'kernel':['rbf']}
# Define svr here
...
svr = svm.SVR()
# Specify cross-validation generator, in this case (10 x 5CV)
cv = RepeatedKFold(n_splits=5, n_repeats=10)
clf = GridSearchCV(estimator=svr, param_grid=param_grid, scoring="neg_mean_squared_error", cv=cv,n_jobs=-1)

# Continue as usual
clf.fit(X_train, y3)



print('Formationen_energy_best_score',clf.best_score_)
print('Formation_energy_best_parameers',clf.best_params_)
a=pd.DataFrame(clf.cv_results_)[['param_gamma','param_C','param_kernel','mean_test_score']]
a.to_csv(sys.argv[4])

