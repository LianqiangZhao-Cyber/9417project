import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgbm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

np.random.seed(42)

# fill in the missing value
train = pd.read_csv('train.csv', index_col='Id')
train = train.fillna(0)
test = pd.read_csv('test.csv', index_col='Id')
test = test.fillna(0)

sample_submission = pd.read_csv('sample_submission.csv')

# training data
y = train["Class"]
X = train.drop(columns='Class')

# standardization, including transfering text data into binary data
preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False),
     make_column_selector(dtype_include=object)),
)
X = preprocessor.fit_transform(X)
test_ = preprocessor.transform(test)

modelLGBM = lgbm.LGBMClassifier(max_depth=3, n_estimators=240, random_state=0, num_leaves=12)
modelLGBM.fit(X, y)
y_pred_train_lgbm = modelLGBM.predict(X)
accuracy_lgbm = accuracy_score(y, y_pred_train_lgbm)
print("Training Accuracy:", accuracy_lgbm * 100)
f1 = f1_score(y, y_pred_train_lgbm)
print("F1 Score:", f1)
positive_class_recall = recall_score(y, y_pred_train_lgbm, pos_label=1)
print("recall_score:", positive_class_recall)

scores = cross_val_score(modelLGBM, X, y, cv=20, scoring='accuracy')
mean_accuracy = scores.mean()
print("Cross val accuracy:", mean_accuracy * 100)


# Log loss of LGBM
def competition_log_loss(y_true, y_pred):
    epsilon = 1e-15  # avoid infinite value
    N = len(y_true)  # sample number

    # compute the total number of 1 and 0
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)

    p_1 = np.clip(y_pred[:, 1], epsilon, 1 - epsilon)
    p_0 = 1 - p_1

    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0)) / N_0
    log_loss_1 = -np.sum(y_true * np.log(p_1)) / N_1

    # return log loss
    return (log_loss_0 + log_loss_1) / 2


y_pred_loss = modelLGBM.predict_proba(X)
print("log loss of training set:", competition_log_loss(y, y_pred_loss))
y_pred_prob_cv = cross_val_predict(modelLGBM, X, y, cv=30, method='predict_proba')
# cv = 10 : log loss of cross val set:  0.42277891725538613
# cv = 20 : log loss of cross val set:  0.41004946878453974
# cv = 30 : log loss of cross val set:  0.4028266750366241
print("log loss of cross val set: ", competition_log_loss(y, y_pred_prob_cv))

# **********************************
# SVM MODEL

svm_model = svm.SVC(kernel='rbf', probability=True, random_state=42, C=1)
svm_model.fit(X, y)
y_pred_train_svm = svm_model.predict(X)
accuracy_svm = accuracy_score(y, y_pred_train_svm)
print("Training Accuracy:", accuracy_svm * 100)
f1 = f1_score(y, y_pred_train_svm)
print("F1 Score:", f1)
positive_class_recall = recall_score(y, y_pred_train_svm, pos_label=1)
print("recall_score:", positive_class_recall)
scores_svm = cross_val_score(svm_model, X, y, cv=40, scoring='accuracy')
mean_accuracy_svm = scores_svm.mean()
print("Cross val accuracy:", mean_accuracy_svm * 100)
predictions = pd.DataFrame(svm_model.predict_proba(test_))
print(predictions)
predictions2 = pd.DataFrame(modelLGBM.predict_proba(test_))
print(predictions2)

blend = (predictions + predictions2) / 2  # simple ensembles like unweighted average

sample_submission['Id'] = test.reset_index()['Id']
sample_submission.class_0 = blend[0]
sample_submission.class_1 = blend[1]
sample_submission.set_index('Id').to_csv('submission.csv')
print(sample_submission)
