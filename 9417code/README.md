# COMP9417 Project: ICR - Identifying Age-Related Conditions

## 1. <a href="https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview" target="_blank"><b>`Kaggle Link:` ICR - Identifying Age-Related Conditions</b></a><br>

## 2. Feature Engineering
The main process:
1. Read the data set from train and test.
2. Classify the features according to the data types prompted in Kaggle.
3. Split or reconstruct some features, and update the data set with train and test dataset.
4. Use different model to train the data and get the best model by using some metrics
5. Modify hyperparameters to get the highest accuracy
6. Predict the test data

## 3. **[`Final Model` in model_MLP.py](model_MLP.py)**
This model is the final one we choosed. </br>
The main idea:
Four structure Mlp in 4MLP_method.py
the mlp3 is the final model which we choose because it has best performance in loss and acc curve and score.
1. make digital standard scale and object type one hot coding in csv
2. build 4 different mlp
3. train and see the loss acc curve
4. use cross validation to get classification_report score
5. select the best mlp
6. the sample process use to predict test csv

## 4. Other Models
### All other models we used in this project are listed below, but were not chosen as the final model.

- ### **[`Randomforest` in model_Randomforest.py](model_Randomforest.py)**

- ### **[`XGboost` in model_XGboost.py](model_XGboost.py)**

- ### **[`LGBM` in model_LGBM&SVM.py](model_LGBM&SVM.py)**

- ### **[`SVM` in model_LGBM&SVM.py](model_LGBM&SVM.py)**

