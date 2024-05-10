#!/usr/bin/env python
# coding: utf-8

# In[59]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[60]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import random
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from xgboost import XGBClassifier
import random
from scipy.stats import uniform, randint
# Suppress any warnings
import warnings
warnings.filterwarnings('ignore')


# ### Load the data

# In[61]:


train = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/train.csv")
test = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/test.csv")
sample_submission = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv')


# In[62]:


train.head(5)


# ### Missing values analysis

# In[63]:


train_missing_values_count = train.isnull().sum().sum()
test_missing_values_count = test.isnull().sum().sum()
print("The number of train data missing values: ", train_missing_values_count)
print("The number of test data missing values: ", test_missing_values_count)


# In[64]:


train.columns[train.isnull().sum() > 0]


# ### Filling missing values by mean

# In[65]:


cols_fill = ['BQ', 'CB', 'CC', 'DU', 'EL', 'FC', 'FL', 'FS', 'GL']
train[cols_fill] = train[cols_fill].fillna(train[cols_fill].mean())


# ### Check again

# In[66]:


train.columns[train.isnull().sum() > 0]


# In[67]:


train_y = train['Class']
train_x = train.drop(['Id', 'Class'], axis=1)
test = test.drop(columns='Id')
train_x


# ### StandardScaler() & OneHotEncoder()

# In[68]:


preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False),
     make_column_selector(dtype_include=object)),
)
X = preprocessor.fit_transform(train_x)
test_ = preprocessor.transform(test)


# In[69]:


y = train_y
y 


# ### Log loss

# In[70]:


import numpy as np

def model_log_loss(y_true, y_pred):
    # Count the number of samples with class 0 and class 1
    N_0 = np.sum(1 - y_true)  # Number of samples with class 0 (negative class)
    N_1 = np.sum(y_true)      # Number of samples with class 1 (positive class)
    
    # Clip the predicted probabilities to avoid numerical instability
    # Clip values between 1e-15 and 1 - 1e-15
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Predicted probabilities of class 1
    p_0 = 1 - p_1                          # Predicted probabilities of class 0
    
    # Calculate the log loss for class 0 and class 1
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0)) / N_0  # Log loss for class 0
    log_loss_1 = -np.sum(y_true * np.log(p_1)) / N_1      # Log loss for class 1
    
    # Average the log losses for both classes and return the result
    return (log_loss_0 + log_loss_1) / 2


# ### XGBOOST MODEL

# In[71]:


# XGBoost classifier
modelXGB = xgb.XGBClassifier(
    max_depth=5,            
    n_estimators=200,       
    random_state=0,         
)
# Train the model
modelXGB.fit(X, y)
y_pred_train_xgb = modelXGB.predict(X)

# calculate accuracy
accuracy_xgb = accuracy_score(y, y_pred_train_xgb)
print("Training Accuracy:", accuracy_xgb * 100)


# cross-validation
scores = cross_val_score(modelXGB, X, y, cv=20, scoring='accuracy')
mean_accuracy = scores.mean()
print("Cross val accuracy:", mean_accuracy * 100)


# calculate F1-score and Recall score on the training set
f1score_train = f1_score(y, y_pred_train_xgb)
recall_train = recall_score(y, y_pred_train_xgb)

print("Training F1-score:", f1score_train)
print("Training Recall:", recall_train)

# cross-validation to calculate mean F1-score and Recall score
f1_scores = cross_val_score(modelXGB, X, y, cv=20, scoring='f1')
recall_scores = cross_val_score(modelXGB, X, y, cv=20, scoring='recall')

mean_f1_score = f1_scores.mean()
mean_recall_score = recall_scores.mean()

print("Cross val F1-score:", mean_f1_score)
print("Cross val Recall:", mean_recall_score)


# In[72]:


# XGBoost classifier
modelXGB = xgb.XGBClassifier(
    max_depth=5,            
    n_estimators=200,       
    random_state=0,         
)
modelXGB.fit(X, y)
y_pred_train_xgb = modelXGB.predict(X)

# Calculate the accuracy on the training set
accuracy_train = accuracy_score(y, y_pred_train_xgb)
print("Training Accuracy:", accuracy_train * 100)

# Calculate Precision, Recall, and F1-score on the training set
precision_train = precision_score(y, y_pred_train_xgb)
recall_train = recall_score(y, y_pred_train_xgb)
f1score_train = f1_score(y, y_pred_train_xgb)

print("Training Precision:", precision_train)
print("Training Recall:", recall_train)
print("Training F1-score:", f1score_train)

# cross-validation
y_pred_prob_cv = cross_val_predict(modelXGB, X, y, cv=20, method='predict_proba')
y_pred_cv = modelXGB.predict(X)

print("================================================================================================")
# extract the predicted probabilities
positive_prob_cv = y_pred_prob_cv[:, 1]

# set a threshold
threshold = 0.2
y_pred_cv = (positive_prob_cv >= threshold).astype(int)

# calculate accuracy, Precision, Recall, and F1-score on the cross-validation set
accuracy_cv = accuracy_score(y, y_pred_cv)
precision_cv = precision_score(y, y_pred_cv)
recall_cv = recall_score(y, y_pred_cv)
f1score_cv = f1_score(y, y_pred_cv)

print("Cross val Accuracy:", accuracy_cv * 100)
print("Cross val Precision:", precision_cv)
print("Cross val Recall:", recall_cv)
print("Cross val F1-score:", f1score_cv)

