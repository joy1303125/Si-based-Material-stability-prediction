#!/usr/bin/env python
# coding: utf-8

# In[283]:


import joblib
from sklearn.model_selection import train_test_split
import numpy as np

import pandas as pd

from sklearn.model_selection import GridSearchCV, RepeatedKFold
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
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV


# # Formation_enegry_per_atom

# In[377]:


df=pd.read_csv('Xrd_MP_train.csv')


df


# In[378]:



X_train=df.iloc[:,14:]



X_train

y_train=df['formation_energy_per_atom']



y_train
df=pd.read_csv('xrd_test_mp.csv')
X_test=df.iloc[:,14:]



X_test
y_test=df['formation_energy_per_atom']



y_test


# # grid_search

# In[104]:


#Formationen_energy_best_score -0.017058333895247656
#Formation_energy_best_parameers {'C': 1, 'epsilon': 0.05, 'gamma': 0.01, 'kernel': 'rbf'}


# In[105]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# In[106]:


model = svm.SVR(C= 1,kernel='rbf',gamma=0.01,epsilon=0.05)
from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[108]:


Xrd_grid_f= model.predict(X_test_normalized)
Xrd_grid_f


# In[109]:


y_test_f=np.array(y_test)
y_test_f


# In[110]:


import math
num=0.018 
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[ ]:





# # bayes_search_Cv

# In[76]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)
params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[31]:





# In[111]:


model = svm.SVR(C= 103.21443810934242,kernel='rbf',gamma=0.0009618994322418767,epsilon=0.07643563515286977)
from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[78]:


import math
num=0.016
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[112]:


Xrd_bayes_f= model.predict(X_test_normalized)
Xrd_bayes_f


# In[113]:


y_test_f=np.array(y_test)
y_test_f


# In[ ]:





# # Packing_fraction

# In[114]:


df=pd.read_csv('Xrd_MP_train.csv')


df.head(2)


# In[115]:


X_train=df.iloc[:,14:]



X_train

y_train=df['packing fraction']



y_train


# In[116]:


df=pd.read_csv('xrd_test_mp.csv')


df.head(2)


# In[117]:



X_test=df.iloc[:,14:]



X_test
y_test=df['packing fraction']



y_test


# In[118]:



from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# # grid_search

# In[ ]:


#Packing_fraction_best_score -0.0016600900514900257
#Packing_fraction_best_parameers {'C': 1, 'epsilon': 0.005, 'gamma': 0.1, 'kernel': 'rbf'}


# In[119]:


model = svm.SVR(C= 1,kernel='rbf',gamma= 0.01,epsilon=0.005)


# In[120]:


from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[121]:


import math
num=0.002
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[122]:


Xrd_grid_p= model.predict(X_test_normalized)
Xrd_grid_p


# In[123]:


y_test_p=np.array(y_test)
y_test_p


# In[ ]:





# In[ ]:





# # bayes_search

# In[71]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[124]:


model = svm.SVR(C= 600.0,kernel='rbf',gamma= 2.0653852486217225e-05,epsilon=3.0046180388205598e-06)


# In[125]:


from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[126]:


model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[127]:


import math
num=0.001
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[128]:


Xrd_bayes_p= model.predict(X_test_normalized)
Xrd_bayes_p


# In[129]:


y_test_p=np.array(y_test)
y_test_p


# In[ ]:





# # Gibbs free energy

# In[379]:


df=pd.read_csv('Xrd_MP_train.csv')


df


# In[380]:


X_train=df.iloc[:,14:]



X_train
y_train=df['energy_per_atom']


y_train


# In[488]:


df=pd.read_csv('xrd_test_mp.csv')


df.head(2)


# In[489]:



X_test=df.iloc[:,14:]



X_test
y_test=df['energy_per_atom']




y_test


# In[490]:



from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# # grid_search

# In[385]:


#Energy_best_score -0.04519640722662908
#Energy_best_parmeters {'C': 21, 'epsilon': 0.05, 'gamma': 0.1, 'kernel': 'rbf'}


# In[491]:


model = svm.SVR(C= 51,kernel='rbf',gamma=0.0001 ,epsilon=5e-05)


# In[492]:


from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[388]:


import math
num=0.053
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[291]:


Xrd_grid_e= model.predict(X_test_normalized)
Xrd_grid_e


# In[246]:


y_test_e=np.array(y_test)
y_test_e


# In[ ]:





# In[ ]:





# # bayes_search

# In[80]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[140]:


model = svm.SVR(C= 600.0,kernel='rbf',gamma= 1.3827406518289601e-06,epsilon=1.5630548520243973e-06)
from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[141]:


import math
num=0.533
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[142]:


Xrd_bayes_e= model.predict(X_test_normalized)
Xrd_bayes_e


# In[143]:


y_test_e=np.array(y_test)
y_test_e


# # Sine coulomb

# # formation_energy

# In[144]:


df=pd.read_csv('Sine_MP_train.csv')


df


# In[145]:


X_train=df.iloc[:,14:]



X_train
y_train=df['formation_energy_per_atom']


y_train


# In[146]:


df=pd.read_csv('sine_mp_test.csv')


df


# In[147]:



X_test=df.iloc[:,14:]



X_test
y_test=df['formation_energy_per_atom']




y_test


# In[148]:



from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# # grid_search

# In[149]:


#Formationen_energy_best_score -0.017058333895247656
#Formation_energy_best_parameers {'C': 1, 'epsilon': 0.05, 'gamma': 0.01, 'kernel': 'rbf'}


# In[150]:


model = svm.SVR(C= 1,kernel='rbf',gamma= 0.01,epsilon= 0.05)
from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[443]:


import math
num=0.017
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[151]:


Sine_grid_f= model.predict(X_test_normalized)
Sine_grid_f


# In[ ]:





# # bayes_search

# In[90]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[152]:


model = svm.SVR(C= 1.7386371188983987,kernel='rbf',gamma= 0.005273126614329326,epsilon= 1.7200460022774906e-05)
from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[397]:


import math
num=0.018
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[153]:


Sine_bayes_f= model.predict(X_test_normalized)
Sine_bayes_f


# # packing_fraction

# In[154]:


df=pd.read_csv('Sine_MP_train.csv')


df


# In[155]:


df=pd.read_csv('Sine_MP_train.csv')


df
X_train=df.iloc[:,14:]



X_train
y_train=df['packing fraction']


y_train


# In[156]:


df=pd.read_csv('sine_mp_test.csv')


df

X_test=df.iloc[:,14:]



X_test
y_test=df['packing fraction']




y_test


# In[157]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# # grid_search

# In[86]:


#Packing_fraction_best_score -0.0016600900514900257
#Packing_fraction_best_parameers {'C': 1, 'epsilon': 0.005, 'gamma': 0.1, 'kernel': 'rbf'}


# In[158]:


model = svm.SVR(C=1,kernel='rbf',gamma= 0.1,epsilon=0.005)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[159]:


Sine_grid_p= model.predict(X_test_normalized)
Sine_grid_p


# In[453]:


import math
num=0.002
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[ ]:





# In[ ]:





# # bayes_search

# In[102]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[160]:


model = svm.SVR(C= 0.9688779005626252,kernel='rbf',gamma= 0.08872158377899908,epsilon=0.023726340977538694)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[161]:


import math
num=0.002
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[162]:


Sine_bayes_p= model.predict(X_test_normalized)
Sine_bayes_p


# # Gibbs free energy

# In[247]:


df=pd.read_csv('Sine_MP_train.csv')


df
X_train=df.iloc[:,14:]



X_train
y_train=df['energy_per_atom']


y_train


# In[248]:


df=pd.read_csv('sine_mp_test.csv')


df

X_test=df.iloc[:,14:]



X_test
y_test=df['energy_per_atom']




y_test


# In[249]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# # grid_search

# In[ ]:


#Energy_best_score -0.04519640722662908
#Energy_best_parmeters {'C': 21, 'epsilon': 0.05, 'gamma': 0.1, 'kernel': 'rbf'}


# In[166]:


model = svm.SVR(C= 21,kernel='rbf',gamma=0.1,epsilon=0.05)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[458]:


import math
num=0.047
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[167]:


Sine_grid_e= model.predict(X_test_normalized)
Sine_grid_e


# # bayes_search

# In[108]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[168]:


model = svm.SVR(C= 4.980365512457193,kernel='rbf',gamma=0.15924525963337247,epsilon=0.020897548846050832)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[395]:


import math
num=0.045
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[169]:


Sine_bayes_e= model.predict(X_test_normalized)
Sine_bayes_e


# # Orbital

# In[372]:


df=pd.read_csv('orbital_MP_train.csv')


df


# In[373]:


df=pd.read_csv('orbital_MP_train.csv')


df
X_train=df.iloc[:,14:]



X_train
y_train=df['formation_energy_per_atom']


y_train


# In[374]:


df=pd.read_csv('Orbital_mp_test.csv')


df

X_test=df.iloc[:,14:]



X_test
y_test=df['formation_energy_per_atom']




y_test


# In[375]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# # grid_search

# In[259]:


model = svm.SVR(C= 201,kernel='rbf',gamma=0.01,epsilon=0.005)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[16]:


import math
num=0.015
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[177]:


Ofm_grid_f= model.predict(X_test_normalized)
Ofm_grid_f


# # bayes_search

# In[126]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[178]:


model = svm.SVR(C= 5.641744052941527,kernel='rbf',gamma=0.5048097231040398,epsilon=0.007469665649787087)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[394]:


import math
num=0.016
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[179]:


Ofm_bayes_f= model.predict(X_test_normalized)
Ofm_bayes_f


# In[ ]:





# In[ ]:





# In[ ]:





# # packing_fraction

# In[260]:


df=pd.read_csv('orbital_MP_train.csv')


df
X_train=df.iloc[:,14:]



X_train
y_train=df['packing fraction']



y_train


# In[261]:


df=pd.read_csv('Orbital_mp_test.csv')


df

X_test=df.iloc[:,14:]



X_test
y_test=df['packing fraction']





y_test


# In[262]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# # grid_search

# In[264]:


model = svm.SVR(C=1 ,kernel='rbf',gamma=0.01,epsilon=5e-05)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[265]:


import math
num=0.001
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[184]:


Ofm_grid_p= model.predict(X_test_normalized)
Ofm_grid_p


# # bayes_search

# In[156]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[185]:


model = svm.SVR(C=0.43144359853099673 ,kernel='rbf',gamma=0.007573902847988706,epsilon=0.0002708080470446585)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[35]:


import math
num=0.001
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[186]:


Ofm_bayes_p= model.predict(X_test_normalized)
Ofm_bayes_p


# # gibbs_free_enegry

# In[371]:


df=pd.read_csv('orbital_MP_train.csv')


df
X_train=df.iloc[:,14:]



X_train
y_train=df['energy_per_atom']


y_train


# In[251]:


df=pd.read_csv('Orbital_mp_test.csv')


df

X_test=df.iloc[:,14:]



X_test
y_test=df['energy_per_atom']




y_test


# In[252]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# # grid_search

# In[254]:


model = svm.SVR(C= 11 ,kernel='rbf',gamma=0.01,epsilon=0.0005)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[26]:


import math
num=0.028 
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[191]:


Ofm_grid_e= model.predict(X_test_normalized)
Ofm_grid_e


# In[ ]:





# # bayes_search

# In[160]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[192]:


model = svm.SVR(C= 276.76589641743413 ,kernel='rbf',gamma=0.009431196207320,epsilon=0.008116063341708436)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[24]:


import math
num=0.023
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[193]:


Ofm_bayes_e= model.predict(X_test_normalized)
Ofm_bayes_e


# # plot

# In[266]:


Xrd_grid_f


# In[267]:


Xrd_bayes_f


# In[362]:


plt.figure(figsize=(12,6))
plt.scatter(range(len(Xrd_grid_f)), Xrd_grid_f, color='blue',marker = "v",alpha=0.9,s=400,label='Xrd grid prediction')
plt.scatter(range(len(Xrd_bayes_f)), Xrd_bayes_f, color='green',marker = 's',s=400,alpha=0.7,label='Xrd bayes prediction')
plt.scatter(range(len(y_test_f)), y_test_f, color='red',marker = "p",s=400,alpha=0.9,label='Actual data')
plt.title('',fontname="Times New Roman", size=30,fontweight="bold")
plt.xlabel('Test structure number',fontname="Times New Roman", size=30,fontweight="bold")
plt.ylabel('Formation energy(eV/atom)',fontname="Times New Roman", size=30,fontweight="bold")
ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
ax.axes.xaxis.set_ticklabels([])

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size':30})
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
ax.tick_params(axis='both', which='major', labelsize=20, width=3)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
plt.legend( fontsize=20, bbox_to_anchor=(0.6368, .856), loc='center left', borderaxespad=0)

plt.ylim([-1, 1])
plt.savefig("XRD_formation.png",dpi=1200)


# In[269]:


Sine_grid_f


# In[270]:


Sine_bayes_f


# In[363]:


plt.figure(figsize=(12,6))
plt.scatter(range(len(Sine_grid_f)), Sine_grid_f, color='blue',marker = "v",alpha=0.9,s=400,label='Sine grid prediction')
plt.scatter(range(len(Sine_bayes_f)), Sine_bayes_f, color='green',marker = 's',s=400,alpha=0.7,label='Sine bayes prediction')
plt.scatter(range(len(y_test_f)), y_test_f, color='red',marker = "p",s=400,alpha=0.9,label='Actual data')
plt.title('',fontname="Times New Roman", size=30,fontweight="bold")
plt.xlabel('',fontname="Times New Roman", size=30,fontweight="bold")
plt.ylabel('',fontname="Times New Roman", size=30,fontweight="bold")
ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
ax.axes.xaxis.set_ticklabels([])

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size':30})
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
ax.tick_params(axis='both', which='major', labelsize=20, width=3)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
plt.legend( fontsize=20, bbox_to_anchor=(0.630868, .856), loc='center left', borderaxespad=0)

plt.ylim([-1, 1])
plt.savefig("Sine_formation.png",dpi=1200)


# In[364]:


plt.figure(figsize=(12,6))
plt.scatter(range(len(Ofm_grid_f)), Ofm_grid_f, color='blue',marker = "v",alpha=0.9,s=400,label='Ofm grid prediction')
plt.scatter(range(len(Ofm_bayes_f)), Ofm_bayes_f, color='green',marker = 's',s=400,alpha=0.7,label='Ofm bayes prediction')
plt.scatter(range(len(y_test_f)), y_test_f, color='red',marker = "p",s=400,alpha=0.9,label='Actual data')
plt.title('',fontname="Times New Roman", size=30,fontweight="bold")
plt.xlabel('',fontname="Times New Roman", size=30,fontweight="bold")
plt.ylabel('',fontname="Times New Roman", size=30,fontweight="bold")
ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
ax.axes.xaxis.set_ticklabels([])

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size':30})
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
ax.tick_params(axis='both', which='major', labelsize=20, width=3)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
plt.legend( fontsize=20, bbox_to_anchor=(0.630868, .856), loc='center left', borderaxespad=0)

plt.ylim([-1, 1])
plt.savefig("Ofm_formation.png",dpi=1200)


# In[ ]:





# In[ ]:





# In[273]:


Xrd_bayes_p


# In[274]:


Xrd_grid_p


# In[275]:


y_test_p


# In[365]:


plt.figure(figsize=(12,6))
plt.scatter(range(len(Xrd_grid_p)), Xrd_grid_p, color='blue',marker = "v",alpha=0.9,s=400,label='Xrd grid prediction')
plt.scatter(range(len(Xrd_bayes_p)), Xrd_bayes_p, color='green',marker = 's',s=400,alpha=0.7,label='Xrd bayes prediction')
plt.scatter(range(len(y_test_p)), y_test_p, color='red',marker = "p",s=400,alpha=0.9,label='Actual data')
plt.title('',fontname="Times New Roman", size=30,fontweight="bold")
plt.xlabel('',fontname="Times New Roman", size=30,fontweight="bold")
plt.ylabel('',fontname="Times New Roman", size=30,fontweight="bold")
ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
ax.axes.xaxis.set_ticklabels([])

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size':30})
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
ax.tick_params(axis='both', which='major', labelsize=20, width=3)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
plt.legend( fontsize=20, bbox_to_anchor=(0.63952868, .856), loc='center left', borderaxespad=0)

plt.ylim([0, 1])
plt.savefig("XRD_packing.png",dpi=1200)


# In[366]:


plt.figure(figsize=(12,6))
plt.scatter(range(len(Sine_grid_p)), Sine_grid_p, color='blue',marker = "v",alpha=0.9,s=400,label='Sine grid prediction')
plt.scatter(range(len(Sine_bayes_p)), Sine_bayes_p, color='green',marker = 's',s=400,alpha=0.7,label='Sine bayes prediction')
plt.scatter(range(len(y_test_p)), y_test_p, color='red',marker = "p",s=400,alpha=0.9,label='Actual data')
plt.title('',fontname="Times New Roman", size=30,fontweight="bold")
plt.xlabel('',fontname="Times New Roman", size=30,fontweight="bold")
plt.ylabel('',fontname="Times New Roman", size=30,fontweight="bold")
ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
ax.axes.xaxis.set_ticklabels([])

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size':30})
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
ax.tick_params(axis='both', which='major', labelsize=20, width=3)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
plt.legend( fontsize=20, bbox_to_anchor=(0.630468, .856), loc='center left', borderaxespad=0)

plt.ylim([0, 1])
plt.savefig("Sine_packing.png",dpi=1200)


# In[367]:


plt.figure(figsize=(12,6))
plt.scatter(range(len(Ofm_grid_p)),Ofm_grid_p, color='blue',marker = "v",alpha=0.9,s=400,label='Ofm grid prediction')
plt.scatter(range(len(Ofm_bayes_p)), Ofm_bayes_p, color='green',marker = 's',s=400,alpha=0.7,label='Ofm bayes prediction')
plt.scatter(range(len(y_test_p)), y_test_p, color='red',marker = "p",s=400,alpha=0.9,label='Actual data')
plt.title(' ',fontname="Times New Roman", size=30,fontweight="bold")
plt.xlabel('',fontname="Times New Roman", size=30,fontweight="bold")
plt.ylabel('',fontname="Times New Roman", size=30,fontweight="bold")
ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
ax.axes.xaxis.set_ticklabels([])

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size':30})
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
ax.tick_params(axis='both', which='major', labelsize=20, width=3)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
plt.legend( fontsize=20, bbox_to_anchor=(0.630468, .856), loc='center left', borderaxespad=0)

plt.ylim([0, 1])
plt.savefig("OFM_packing.png",dpi=1200)


# In[ ]:





# In[279]:


Xrd_grid_e


# In[368]:


plt.figure(figsize=(12,6))
plt.scatter(range(len(Xrd_grid_e)), Xrd_grid_e, color='blue',marker = "v",alpha=0.9,s=400,label='Xrd grid prediction')
plt.scatter(range(len(Xrd_bayes_e)), Xrd_bayes_e, color='green',marker = 's',s=400,alpha=0.7,label='Xrd bayes prediction')
plt.scatter(range(len(y_test_e)), y_test_e, color='red',marker = "p",s=400,alpha=0.9,label='Actual data')
plt.title(' ',fontname="Times New Roman", size=30,fontweight="bold")
plt.xlabel('',fontname="Times New Roman", size=30,fontweight="bold")
plt.ylabel('',fontname="Times New Roman", size=30,fontweight="bold")
ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
ax.axes.xaxis.set_ticklabels([])

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size':30})
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
ax.tick_params(axis='both', which='major', labelsize=20, width=3)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
plt.legend( fontsize=20, bbox_to_anchor=(0.6390168, .856), loc='center left', borderaxespad=0)

plt.ylim([-1, -10])
plt.savefig("XRD_gibbs.png",dpi=1200)


# In[369]:


plt.figure(figsize=(12,6))
plt.scatter(range(len(Sine_grid_e)), Sine_grid_e, color='blue',marker = "v",alpha=0.9,s=400,label='Sine grid prediction')
plt.scatter(range(len(Sine_bayes_e)), Sine_bayes_e, color='green',marker = 's',s=400,alpha=0.7,label='Sine bayes prediction')
plt.scatter(range(len(y_test_e)), y_test_e, color='red',marker = "p",s=400,alpha=0.9,label='Actual data')
plt.title('',fontname="Times New Roman", size=30,fontweight="bold")
plt.xlabel('',fontname="Times New Roman", size=30,fontweight="bold")
plt.ylabel('',fontname="Times New Roman", size=30,fontweight="bold")
ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
ax.axes.xaxis.set_ticklabels([])

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size':30})
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
ax.tick_params(axis='both', which='major', labelsize=20, width=3)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
plt.legend( fontsize=20, bbox_to_anchor=(0.629390168, .856), loc='center left', borderaxespad=0)

plt.ylim([-1, -10])
plt.savefig("Sine_gibbs.png",dpi=1200)


# In[370]:


plt.figure(figsize=(12,6))
plt.scatter(range(len(Ofm_grid_e)), Ofm_grid_e, color='blue',marker = "v",alpha=0.9,s=400,label='Ofm grid prediction')
plt.scatter(range(len(Ofm_bayes_e)), Ofm_bayes_e, color='green',marker = 's',s=400,alpha=0.7,label='Ofm bayes prediction')
plt.scatter(range(len(y_test_e)), y_test_e, color='red',marker = "p",s=400,alpha=0.9,label='Actual data')
plt.title('',fontname="Times New Roman", size=30,fontweight="bold")
plt.xlabel('',fontname="Times New Roman", size=30,fontweight="bold")
plt.ylabel('',fontname="Times New Roman", size=30,fontweight="bold")
ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
ax.axes.xaxis.set_ticklabels([])

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size':30})
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
ax.tick_params(axis='both', which='major', labelsize=20, width=3)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontweight('bold')
plt.legend( fontsize=20, bbox_to_anchor=(0.629390168, .856), loc='center left', borderaxespad=0)

plt.ylim([-1, -10])
plt.savefig("ofm_gibbs.png",dpi=1200)


# # Skewness_result

# # XRD

# # Formation_energy

# In[211]:


df=pd.read_csv('Xrd_MP_train.csv')


df.head(3)


# In[212]:


df = df.loc[~((df['Class_name'] == 'Mgbased') & (df['e_above_hull'] >0.2))]
df


# In[213]:


df = df.reset_index(drop=True)


# In[214]:


df


# In[215]:



X_train=df.iloc[:,14:]



X_train

y_train=df['formation_energy_per_atom']



y_train
df=pd.read_csv('xrd_test_mp.csv')
X_test=df.iloc[:,14:]



X_test
y_test=df['formation_energy_per_atom']



y_test


# In[218]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# In[180]:



params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[219]:


model = svm.SVR(C= 43.98139503202982,kernel='rbf',gamma=0.00015760882734503523,epsilon=2.039577492677281e-06)
from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[391]:


import math
num=0.012  
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[ ]:





# # packing_fraction_prediction

# In[182]:


df=pd.read_csv('Xrd_MP_train.csv')


df.head(3)


# In[184]:


df = df.loc[~((df['Class_name'] == 'Mgbased') & (df['e_above_hull'] >0.2))]
df.head(3)


# In[185]:


df = df.reset_index(drop=True)


# In[186]:


X_train=df.iloc[:,14:]



X_train

y_train=df['packing fraction']



y_train
df=pd.read_csv('xrd_test_mp.csv')
X_test=df.iloc[:,14:]



X_test
y_test=df['packing fraction']



y_test


# In[187]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)
params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[188]:


model = svm.SVR(C= 600,kernel='rbf',gamma=0.00014578070263138504,epsilon= 0.005354011286248393)
from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[390]:


import math
num=0.001  
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[ ]:





# # energy_per_atom

# In[189]:


df=pd.read_csv('Xrd_MP_train.csv')


df.head(3)


# In[190]:


df = df.loc[~((df['Class_name'] == 'Mgbased') & (df['e_above_hull'] >0.2))]
df.head(3)


# In[191]:


df = df.reset_index(drop=True)


# In[192]:


X_train=df.iloc[:,14:]



X_train

y_train=df['energy_per_atom']



y_train
df=pd.read_csv('xrd_test_mp.csv')
X_test=df.iloc[:,14:]



X_test
y_test=df['energy_per_atom']



y_test


# In[193]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)
params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[195]:


model = svm.SVR(C= 145.08096558282043,kernel='rbf',gamma=0.00017599837794165854,epsilon=6.748768906729976e-06)
from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X0.12_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[387]:


import math
num=0.062 
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # SIne_coulmb_matrix
# formation_energy

# In[197]:


df=pd.read_csv('Sine_MP_train.csv')


df.head(3)


# In[198]:


df = df.loc[~((df['Class_name'] == 'Mgbased') & (df['e_above_hull'] >0.2))]
df.head(3)


# In[199]:


df = df.reset_index(drop=True)


# In[200]:


X_train=df.iloc[:,14:]



X_train
y_train=df['formation_energy_per_atom']


y_train


# In[ ]:





# In[201]:


df=pd.read_csv('sine_mp_test.csv')


X_test=df.iloc[:,14:]



X_test
y_test=df['formation_energy_per_atom']




y_test


# In[202]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)
params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[203]:


model = svm.SVR(C= 0.5813834029267732,kernel='rbf',gamma= 0.06100124458773761,epsilon=0.011554748279473729)
from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[386]:


import math
num=0.015 
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# # packing_fraction

# In[205]:


df=pd.read_csv('Sine_MP_train.csv')


df.head(3)


# In[206]:


df = df.loc[~((df['Class_name'] == 'Mgbased') & (df['e_above_hull'] >0.2))]
df.head(3)
df = df.reset_index(drop=True)


# In[207]:


df = df.reset_index(drop=True)


# In[208]:


X_train=df.iloc[:,14:]



X_train
y_train=df['packing fraction']


y_train


# In[209]:


df=pd.read_csv('sine_mp_test.csv')


df

X_test=df.iloc[:,14:]



X_test
y_test=df['packing fraction']




y_test


# In[210]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# In[211]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[212]:


model = svm.SVR(C= 0.5341223110608262,kernel='rbf',gamma= 0.19263136370227923,epsilon=0.01093439308161429)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[385]:


import math
num=0.002
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# # Gibbs free energy
# 

# In[232]:


df=pd.read_csv('Sine_MP_train.csv')


df.head(3)


# In[233]:


df = df.loc[~((df['Class_name'] == 'Mgbased') & (df['e_above_hull'] >0.2))]
df.head(3)
df = df.reset_index(drop=True)


# In[ ]:





# In[234]:


X_train=df.iloc[:,14:]



X_train
y_train=df['energy_per_atom']


y_train


# In[235]:


df=pd.read_csv('sine_mp_test.csv')


df

X_test=df.iloc[:,14:]



X_test
y_test=df['energy_per_atom']




y_test


# In[236]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# In[221]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[384]:


def to_exponential(num):
    return '{:e}'.format(num)
to_exponential(0.00019996967578188855)


# In[237]:


model = svm.SVR(C= 600.0,kernel='rbf',gamma=  2.7822719111549485,epsilon=0.00019996967578188855)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[383]:


import math
num=0.076 
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# # Orbital_field_matrix
##formation_energy_per_atom
# In[4]:


df=pd.read_csv('orbital_MP_train.csv')


df.head(3)


# In[225]:


df = df.loc[~((df['Class_name'] == 'Mgbased') & (df['e_above_hull'] >0.2))]
df.head(3)
df = df.reset_index(drop=True)


# In[226]:


X_train=df.iloc[:,14:]



X_train
y_train=df['formation_energy_per_atom']


y_train


# In[228]:


df=pd.read_csv('Orbital_mp_test.csv')


df

X_test=df.iloc[:,14:]



X_test
y_test=df['formation_energy_per_atom']




y_test


# In[229]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# In[230]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[231]:


model = svm.SVR(C= 2.6774936135466283,kernel='rbf',gamma=0.20690210143774546,epsilon=7.0220733662899515e-06)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[381]:


import math
num=0.014
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# # packing fraction

# In[238]:


df=pd.read_csv('orbital_MP_train.csv')


df.head(3)


# In[239]:


df = df.loc[~((df['Class_name'] == 'Mgbased') & (df['e_above_hull'] >0.2))]
df.head(3)
df = df.reset_index(drop=True)


# In[240]:


X_train=df.iloc[:,14:]



X_train
y_train=df['packing fraction']


y_train


# In[241]:


df=pd.read_csv('Orbital_mp_test.csv')


df

X_test=df.iloc[:,14:]



X_test
y_test=df['packing fraction']




y_test


# In[242]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# In[243]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[244]:


model = svm.SVR(C= 7.800253097493293,kernel='rbf',gamma=0.20104384114906393,epsilon=0.01161429762446919)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[382]:


import math
num=0.0010
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# # gibbs_free_enegry
# 

# In[19]:


df=pd.read_csv('orbital_MP_train.csv')


df


# In[373]:


df = df.loc[~((df['Class_name'] == 'Mgbased') & (df['e_above_hull'] >0.2))]
df.head(3)
df = df.reset_index(drop=True)


# In[374]:


X_train=df.iloc[:,14:]



X_train
y_train=df['energy_per_atom']


y_train


# In[375]:


df=pd.read_csv('Orbital_mp_test.csv')


df

X_test=df.iloc[:,14:]



X_test
y_test=df['energy_per_atom']




y_test


# In[376]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# In[377]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[378]:


model = svm.SVR(C= 24.881650683716856,kernel='rbf',gamma=0.00569930333302307,epsilon=1.1295746128956395e-06)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[379]:


import math
num=0.033 
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # outliers

# # formation_energy

# In[604]:


df=pd.read_csv('Xrd_MP_train.csv')


df.head(3)


# In[605]:



# calculate the median and the median absolute deviation (MAD) of the data
median = df['formation_energy_per_atom'].median()
MAD = df['formation_energy_per_atom'].mad()

# calculate the modified Z-scores for each data point
df['mod_z_score'] = 0.6745 * (df['formation_energy_per_atom'] - median) / MAD

# set a threshold value for the modified Z-score
mod_z_score_threshold = 3.5

# identify data points with modified Z-score > threshold
outliers = df[df['mod_z_score'].abs() > mod_z_score_threshold]

# remove rows corresponding to outliers
df = df[df['mod_z_score'].abs() <= mod_z_score_threshold]
del df['mod_z_score']


# In[606]:


X_train=df.iloc[:,14:]



X_train

y_train=df['formation_energy_per_atom']



y_train
df=pd.read_csv('xrd_test_mp.csv')
X_test=df.iloc[:,14:]



X_test
y_test=df['formation_energy_per_atom']



y_test


# In[607]:


X_train.shape


# In[273]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)
params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[274]:


model = svm.SVR(C= 1.0418096141855044,kernel='rbf',gamma= 0.0012955865251206558,epsilon=0.007072084736280741)
from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[603]:


import math
num=0.002
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# # packing_fraction

# In[608]:


df=pd.read_csv('Xrd_MP_train.csv')


df.head(3)


# In[609]:


df.shape


# In[ ]:





# In[ ]:


We also
removed 18 other outliers by applying the modified Z score method 83 shown in Figure S1.


# In[ ]:





# In[610]:



# calculate the median and the median absolute deviation (MAD) of the data
median = df['packing fraction'].median()
MAD = df['packing fraction'].mad()

# calculate the modified Z-scores for each data point
df['mod_z_score'] = 0.6745 * (df['packing fraction'] - median) / MAD

# set a threshold value for the modified Z-score
mod_z_score_threshold = 3.5

# identify data points with modified Z-score > threshold
outliers = df[df['mod_z_score'].abs() > mod_z_score_threshold]

# remove rows corresponding to outliers
df = df[df['mod_z_score'].abs() <= mod_z_score_threshold]
del df['mod_z_score']


# In[611]:


X_train=df.iloc[:,14:]



X_train

y_train=df['packing fraction']



y_train
df=pd.read_csv('xrd_test_mp.csv')
X_test=df.iloc[:,14:]



X_test
y_test=df['packing fraction']



y_test


# In[592]:


X_train


# In[368]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)
params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[370]:


model = svm.SVR(C= 257.4557517434784,kernel='rbf',gamma= 4.304821957575336e-05,epsilon=0.0003457952185439735)
from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[364]:


import math
num=0.007
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[ ]:





# # Energy_per_atom

# In[616]:


df=pd.read_csv('Xrd_MP_train.csv')


df.head(3)


# In[617]:



# calculate the median and the median absolute deviation (MAD) of the data
median = df['energy_per_atom'].median()
MAD = df['energy_per_atom'].mad()

# calculate the modified Z-scores for each data point
df['mod_z_score'] = 0.6745 * (df['energy_per_atom'] - median) / MAD

# set a threshold value for the modified Z-score
mod_z_score_threshold = 3.5

# identify data points with modified Z-score > threshold
outliers = df[df['mod_z_score'].abs() > mod_z_score_threshold]

# remove rows corresponding to outliers
df = df[df['mod_z_score'].abs() <= mod_z_score_threshold]
del df['mod_z_score']


# In[618]:


X_train=df.iloc[:,14:]



X_train

y_train=df['energy_per_atom']



y_train
df=pd.read_csv('xrd_test_mp.csv')
X_test=df.iloc[:,14:]



X_test
y_test=df['energy_per_atom']



y_test


# In[619]:


X_train


# In[ ]:





# In[285]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)
params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[286]:


model = svm.SVR(C=  42.574332597720144,kernel='rbf',gamma= 0.0005205878405169264,epsilon=6.846842124401742e-05)
from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[363]:


import math
num=0.039
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[ ]:





# # SIne_coulmb_matrix
# 

# In[287]:


df=pd.read_csv('Sine_MP_train.csv')


df.head(3)


# In[303]:



# calculate the median and the median absolute deviation (MAD) of the data
median = df['formation_energy_per_atom'].median()
MAD = df['formation_energy_per_atom'].mad()

# calculate the modified Z-scores for each data point
df['mod_z_score'] = 0.6745 * (df['formation_energy_per_atom'] - median) / MAD

# set a threshold value for the modified Z-score
mod_z_score_threshold = 3.5

# identify data points with modified Z-score > threshold
outliers = df[df['mod_z_score'].abs() > mod_z_score_threshold]

# remove rows corresponding to outliers
df = df[df['mod_z_score'].abs() <= mod_z_score_threshold]
del df['mod_z_score']


# In[305]:


X_train=df.iloc[:,14:]



X_train

y_train=df['formation_energy_per_atom']



y_train
df=pd.read_csv('sine_mp_test.csv')
X_test=df.iloc[:,14:]



X_test
y_test=df['formation_energy_per_atom']



y_test


# In[306]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)
params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[307]:


model = svm.SVR(C= 0.07362898250633315,kernel='rbf',gamma= 0.6941249236123415,epsilon= 1.4875876867513227e-05)
from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[362]:


import math
num=0.003
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# # packing-fraction

# In[309]:


df=pd.read_csv('Sine_MP_train.csv')


df.head(3)


# In[310]:



# calculate the median and the median absolute deviation (MAD) of the data
median = df['packing fraction'].median()
MAD = df['packing fraction'].mad()

# calculate the modified Z-scores for each data point
df['mod_z_score'] = 0.6745 * (df['packing fraction'] - median) / MAD

# set a threshold value for the modified Z-score
mod_z_score_threshold = 3.5

# identify data points with modified Z-score > threshold
outliers = df[df['mod_z_score'].abs() > mod_z_score_threshold]

# remove rows corresponding to outliers
df = df[df['mod_z_score'].abs() <= mod_z_score_threshold]
del df['mod_z_score']


# In[311]:


X_train=df.iloc[:,14:]



X_train

y_train=df['packing fraction']



y_train

df=pd.read_csv('sine_mp_test.csv')

X_test=df.iloc[:,14:]



X_test
y_test=df['packing fraction']



y_test


# In[312]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)
params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[313]:


model = svm.SVR(C= 0.5264062290050722,kernel='rbf',gamma=0.16071819878688814,epsilon=0.011171925562928915)
from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[361]:


import math
num=0.001
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# # energy_per_atom

# In[314]:


df=pd.read_csv('Sine_MP_train.csv')


df.head(3)


# In[315]:


# calculate the median and the median absolute deviation (MAD) of the data
median = df['energy_per_atom'].median()
MAD = df['energy_per_atom'].mad()

# calculate the modified Z-scores for each data point
df['mod_z_score'] = 0.6745 * (df['energy_per_atom'] - median) / MAD

# set a threshold value for the modified Z-score
mod_z_score_threshold = 3.5

# identify data points with modified Z-score > threshold
outliers = df[df['mod_z_score'].abs() > mod_z_score_threshold]

# remove rows corresponding to outliers
df = df[df['mod_z_score'].abs() <= mod_z_score_threshold]
del df['mod_z_score']


# In[317]:


X_train=df.iloc[:,14:]



X_train

y_train=df['energy_per_atom']



y_train
df=pd.read_csv('sine_mp_test.csv')
X_test=df.iloc[:,14:]



X_test
y_test=df['energy_per_atom']



y_test


# In[318]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)
params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[319]:


model = svm.SVR(C= 428.08634129433347,kernel='rbf',gamma=1e-06,epsilon=1.6336559418596864e-06)
from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[359]:


import math
num=0.550 
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[ ]:





# In[ ]:





# # orbital_field_matrix

# In[324]:


df=pd.read_csv('orbital_MP_train.csv')



df.head(3)


# In[325]:


# calculate the median and the median absolute deviation (MAD) of the data
median = df['formation_energy_per_atom'].median()
MAD = df['formation_energy_per_atom'].mad()

# calculate the modified Z-scores for each data point
df['mod_z_score'] = 0.6745 * (df['formation_energy_per_atom'] - median) / MAD

# set a threshold value for the modified Z-score
mod_z_score_threshold = 3.5

# identify data points with modified Z-score > threshold
outliers = df[df['mod_z_score'].abs() > mod_z_score_threshold]

# remove rows corresponding to outliers
df = df[df['mod_z_score'].abs() <= mod_z_score_threshold]
del df['mod_z_score']


# In[326]:


X_train=df.iloc[:,14:]



X_train
y_train=df['formation_energy_per_atom']


y_train


# In[327]:


df=pd.read_csv('Orbital_mp_test.csv')


df

X_test=df.iloc[:,14:]



X_test
y_test=df['formation_energy_per_atom']




y_test


# In[328]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# In[329]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[330]:


model = svm.SVR(C= 0.48792777159998413 ,kernel='rbf',gamma=0.17556829929880052,epsilon=0.00036909180591737694)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[358]:


import math
num=0.002 
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[ ]:





# In[ ]:





# In[331]:


##packing_fraction


# In[333]:


df=pd.read_csv('orbital_MP_train.csv')



df.head(3)


# In[334]:


# calculate the median and the median absolute deviation (MAD) of the data
median = df['packing fraction'].median()
MAD = df['packing fraction'].mad()

# calculate the modified Z-scores for each data point
df['mod_z_score'] = 0.6745 * (df['packing fraction'] - median) / MAD

# set a threshold value for the modified Z-score
mod_z_score_threshold = 3.5

# identify data points with modified Z-score > threshold
outliers = df[df['mod_z_score'].abs() > mod_z_score_threshold]

# remove rows corresponding to outliers
df = df[df['mod_z_score'].abs() <= mod_z_score_threshold]
del df['mod_z_score']


# In[335]:


X_train=df.iloc[:,14:]



X_train
y_train=df['packing fraction']



y_train


# In[336]:


df=pd.read_csv('Orbital_mp_test.csv')


df

X_test=df.iloc[:,14:]



X_test
y_test=df['packing fraction']





y_test


# In[337]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# In[338]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[339]:


model = svm.SVR(C=10.891311958493649,kernel='rbf',gamma=0.0974200793943691,epsilon=0.00011627511529735553)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[340]:


##gibbs_free_energy


# In[347]:


df=pd.read_csv('orbital_MP_train.csv')



df.head(3)


# In[348]:


# calculate the median and the median absolute deviation (MAD) of the data
median = df['energy_per_atom'].median()
MAD = df['energy_per_atom'].mad()

# calculate the modified Z-scores for each data point
df['mod_z_score'] = 0.6745 * (df['energy_per_atom'] - median) / MAD

# set a threshold value for the modified Z-score
mod_z_score_threshold = 3.5

# identify data points with modified Z-score > threshold
outliers = df[df['mod_z_score'].abs() > mod_z_score_threshold]

# remove rows corresponding to outliers
df = df[df['mod_z_score'].abs() <= mod_z_score_threshold]
del df['mod_z_score']


# In[350]:


X_train=df.iloc[:,14:]



X_train
y_train=df['energy_per_atom']



y_train


# In[351]:


df=pd.read_csv('Orbital_mp_test.csv')


df

X_test=df.iloc[:,14:]



X_test
y_test=df['energy_per_atom']





y_test


# In[352]:


from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit scaler on the training data
scaler.fit(X_train)

# apply the scaler to both the training and test data
X_train_normalized = scaler.transform(X_train)

X_test_normalized = scaler.transform(X_test)


# In[353]:


params = dict()
params['C'] = (1e-6, 600.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['epsilon']=(1e-6, 10.0, 'log-uniform')

params['kernel'] = ['rbf']
# define evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# define the search
search = BayesSearchCV(estimator=svm.SVR(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X_train_normalized, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)


# In[354]:


model = svm.SVR(C=87.92553169828344,kernel='rbf',gamma=0.01231442940806244,epsilon=2.358740950185116e-06)

from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# compute cross validation scores for random forest model
scores = cross_val_score(model,X_train_normalized, y_train, scoring= 'neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train_normalized, y_train)
pred = model.predict(X_train_normalized)
rmse = np.sqrt(MSE(y_train, pred))
print("RMSE : % f" %(rmse))
pred = model.predict(X_test_normalized)
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE1 : % f" %(rmse))


# In[620]:


import math
num=-mean(scores)
def square_root(num):
    print(math.sqrt(num))
square_root(num)


# In[ ]:




