import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, log_loss, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import shap
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector


train_df =pd.read_csv('D:/9417/project/icr-identify-age-related-conditions/train.csv')
eval_df =pd.read_csv('D:/9417/project/icr-identify-age-related-conditions/test.csv')
greeks_df = pd.read_csv('D:/9417/project/icr-identify-age-related-conditions/greeks.csv')

print(train_df.describe())

print(train_df["Class"])

train_df.hist(figsize=(28,14), bins=25, layout=(6,11), grid=False);

for col in ['Alpha', 'Beta', 'Gamma', 'Delta']:
    print(col)
    print(greeks_df[col].value_counts())

greeks_df[['Alpha', 'Beta', 'Gamma', 'Delta']].value_counts()

train_corr_data = train_df.corr(method='pearson')

f, ax = plt.subplots(figsize=(14, 12))

sns.heatmap(train_corr_data, ax=ax, cmap='viridis');

print("Top 20 positive correlations with the target")
print(train_corr_data['Class'].sort_values(ascending=False).head(20))

print("Top 20 negative correlations with the target")
print(train_corr_data['Class'].sort_values(ascending=True).head(20))

# Drop Id column
train_df.drop(columns=['Id'], inplace=True)
# Encode EJ_A = 0, EJ_B =1
train_df['EJ'] = train_df['EJ'].replace(to_replace='A', value=0.0)
train_df['EJ'] = train_df['EJ'].replace(to_replace='B', value=1.0)

eval_df_IDs = eval_df['Id']
eval_df.drop(columns=['Id'], inplace=True)
# Encode EJ_A = 0, EJ_B =1
eval_df['EJ'] = eval_df['EJ'].replace(to_replace='A', value=0.0)
eval_df['EJ']= eval_df['EJ'].replace(to_replace='B', value=1.0)

print(train_df.isnull().sum().any())
print(train_df.isnull().sum())

# Imputation of missing values with column median values
train_df['BQ'].fillna(train_df['BQ'].median(), inplace=True)
train_df['CB'].fillna(train_df['CB'].median(), inplace=True)
train_df['CC'].fillna(train_df['CC'].median(), inplace=True)
train_df['DU'].fillna(train_df['DU'].median(), inplace=True)
train_df['EL'].fillna(train_df['EL'].median(), inplace=True)
train_df['FC'].fillna(train_df['FC'].median(), inplace=True)
train_df['FL'].fillna(train_df['FL'].median(), inplace=True)
train_df['FS'].fillna(train_df['FS'].median(), inplace=True)
train_df['GL'].fillna(train_df['GL'].median(), inplace=True)


######################################################################################
# Function to prepare data subsets X_train, y_train, X_test, y_test
def getTrainTestData(df, ts=0.2):
  # Separate predictors from predicted variable
  X = df.iloc[:, 0:56].reset_index(drop=True)
  y = df.iloc[:, 56:57].reset_index(drop=True)
  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=13)
  return X_train, y_train, X_test, y_test
######################################################################################
# Function to calculate model evaluation metrics, returns MAE and R2
def getModelMetrics(model, X, y):
  # Make predictions
  predictions = model.predict(X)
  # Calculate metrics: MAE and R2
  ROC_AUC = round(roc_auc_score(y, predictions),2)
  LOG_LOSS = round(log_loss(y, predictions),2)
  R2 = round(r2_score(y, predictions),2)
  F1 = round(f1_score(y,predictions),2)
  precision = round(precision_score(y, predictions),2)
  recall = round(recall_score(y, predictions),2)
  return ROC_AUC, LOG_LOSS, R2, F1, precision, recall
######################################################################################


# Prepare train and test data
X_train, y_train, X_test, y_test = getTrainTestData(train_df, ts=0.2)
print(X_train.shape, y_train.shape)
print(X_train.sample(5))

# Model parameters
ne = 200
md = 20

# Build and evaluate model
model_RForest = RandomForestClassifier(n_estimators=ne, criterion='gini', random_state=42)
model_RForest.fit(X_train, y_train)
X = train_df.drop(columns='Class')
preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False),
     make_column_selector(dtype_include=object)),
)
X = preprocessor.fit_transform(X)

ROC, LL, R2, f1, precise, reca = getModelMetrics(model_RForest, X_test, y_test)
X = train_df.drop(columns='Class')
y = train_df["Class"]
scores = cross_val_score(model_RForest, X, y, cv=20, scoring='accuracy')
mean_accuracy = scores.mean()
# Print results
print("Random Forest Classifier model")
print('-' * 25)
print(" Learners \t" + str(ne))
print(" ROC AUC \t" + str(ROC))
print(" LOG LOSS \t" + str(LL))
print(" R2 \t\t" + str(R2))
print(" F1 \t\t" + str(f1))
print(" Precision \t\t" + str(precise))
print(" Recall \t\t" + str(reca))
print(" Cross val accuracy:", mean_accuracy * 100)
print('-' * 25)
# Calculate and show Shap Values for the Test Dataset
####################################################

explainer = shap.TreeExplainer(model_RForest)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_size=[8,8], plot_type='bar')
eval_predictions = model_RForest.predict(eval_df)
eval_predictions_probabilities = model_RForest.predict_proba(eval_df)
print(eval_predictions_probabilities)

