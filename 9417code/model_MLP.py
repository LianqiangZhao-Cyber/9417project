import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# from tensorflow import keras
import keras
from keras import layers
# from tensorflow.keras import layers, Input
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from keras.callbacks import EarlyStopping
# from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

'''
Four structure Mlp in 4MLP_method.py
    the mlp3 is the final model which we choose because it has best poerformace in loss and acc curve and score.
    1.make digital standard scale and object type one hot coding in csv
    2.build 4 different mlp
    3.train and see the loss acc curve
    4.use cross validation to get classification_report score
    5.select the best mlp
    6.the sample process use to predict test csv
'''

train_dir = './train.csv'
test_dir = './test.csv'
sample_dir = './sample_submission.csv'

train_csv = pd.read_csv(train_dir)
train_csv.dropna(inplace=True)
train_y = train_csv['Class']
train_csv.drop('Class', axis=1, inplace=True)
train_csv.drop('Id', axis=1, inplace=True)
train_x = train_csv
train_x_object = train_x.select_dtypes(include=['object'])
train_x_num = train_csv.select_dtypes(include=['float64'])

# standard scale for numbers,onehot coding for string
Onehot = OneHotEncoder()
str_features = pd.DataFrame(Onehot.fit_transform(train_x_object).toarray())
Standard = StandardScaler()
num_features = pd.DataFrame(Standard.fit_transform(train_x_num))
x = pd.concat([num_features, str_features], axis=1)
x.columns = range(len(x.columns))

# split into train valid and test
X_train, X_valid, y_train, y_valid = train_test_split(x, train_y, train_size=0.8, test_size=0.2, random_state=11)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.8, random_state=11)


def cross_validate_keras_model(model, X, y, n_splits, epochs, batch_size):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # restore the result
    y_pred_prob_cv = []
    y_true_cv = []

    # loop
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]

        # train
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=0)

        # predict
        y_pred_prob = model.predict(X_test)

        # add to result list
        y_pred_prob_cv.append(y_pred_prob)
        y_true_cv.append(y_test)

    # concat result
    y_pred_prob_cv = np.concatenate(y_pred_prob_cv, axis=0)
    y_true_cv = np.concatenate(y_true_cv, axis=0)

    return y_true_cv, y_pred_prob_cv


def draw_loss_acc(history):
    history_frame = pd.DataFrame(history.history)
    epochs = len(history_frame)

    # create 1x2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # first subplot is loss curve
    axes[0].plot(history_frame['loss'], label='Training Loss')
    axes[0].plot(history_frame['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_xlim([0, epochs])
    axes[0].set_ylim([0, 1])
    axes[0].legend()

    # second subplot is accuracy curve
    axes[1].plot(history_frame['binary_accuracy'], label='Training Accuracy')
    axes[1].plot(history_frame['val_binary_accuracy'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlim([0, epochs])
    axes[1].set_ylim([0, 1])
    axes[1].legend()

    plt.tight_layout()
    return plt


def draw_confusionMatrix(conf_matrix):
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    return plt


early_stopping = EarlyStopping(
    monitor='loss',
    min_delta=0.01,
    patience=20,
    restore_best_weights=True,
)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

'''MLP 1'''
print('----------mlp1----------')
mlp1 = keras.Sequential([
    layers.Dense(56, activation='relu', input_shape=[57]),
    layers.Dense(1, activation='sigmoid')
])
mlp1.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
history1 = mlp1.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=32,callbacks=[early_stopping],verbose=0)

# plot the loss and acc
loss_acc_plot1 = draw_loss_acc(history1)
loss_acc_plot1.savefig('loss_acc_plot1.png')

# cross validation for all data
y_true_cv, y_pred_prob_cv = cross_validate_keras_model(mlp1, x, train_y, n_splits=20, epochs=100, batch_size=32)
y_pred_labels = (y_pred_prob_cv >= 0.5).astype(int)
report = classification_report(y_true_cv, y_pred_labels)
print(report)
conf_matrix = confusion_matrix(y_true_cv, y_pred_labels)
conf_matrix1 = draw_confusionMatrix(conf_matrix)
conf_matrix1.savefig('conf_matrix1.png')

'''MLP 2'''
print('----------mlp2----------')
mlp2 = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=[57]),
    layers.Dense(1, activation='sigmoid')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
mlp2.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
history2 = mlp2.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=32,callbacks=[early_stopping],verbose=0)

# plot the loss and acc
loss_acc_plot2 = draw_loss_acc(history2)
loss_acc_plot2.savefig('loss_acc_plot2.png')

# corss validation
y_true_cv, y_pred_prob_cv = cross_validate_keras_model(mlp2, x, train_y, n_splits=20, epochs=100, batch_size=32)
y_pred_labels = (y_pred_prob_cv >= 0.5).astype(int)
report = classification_report(y_true_cv, y_pred_labels)
print(report)
conf_matrix = confusion_matrix(y_true_cv, y_pred_labels)
conf_matrix2 = draw_confusionMatrix(conf_matrix)
conf_matrix2.savefig('conf_matrix2.png')

'''MLP3'''
print('----------mlp3----------')
mlp3 = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[57]),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
mlp3.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
history3 = mlp3.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, callbacks=[early_stopping],batch_size=32,verbose=0)

# draw loss and acc
loss_acc_plot3 = draw_loss_acc(history3)
loss_acc_plot3.savefig('loss_acc_plot3.png')

# corss validation
y_true_cv, y_pred_prob_cv = cross_validate_keras_model(mlp3, x, train_y, n_splits=20, epochs=100, batch_size=32)
y_pred_labels = (y_pred_prob_cv >= 0.5).astype(int)
report = classification_report(y_true_cv, y_pred_labels)
print(report)
conf_matrix = confusion_matrix(y_true_cv, y_pred_labels)
conf_matrix3 = draw_confusionMatrix(conf_matrix)
conf_matrix3.savefig('conf_matrix3.png')

'''MLP4'''
mlp4 = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[57]),
    layers.Dropout(0.25),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
mlp4.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
history4 = mlp4.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=32,callbacks=[early_stopping],verbose=0)

# draw loss and acc
loss_acc_plot4 = draw_loss_acc(history4)
loss_acc_plot4.savefig('loss_acc_plot4.png')

# corss validation
y_true_cv, y_pred_prob_cv = cross_validate_keras_model(mlp4, x, train_y, n_splits=20, epochs=100, batch_size=32)
y_pred_labels = (y_pred_prob_cv >= 0.5).astype(int)
report = classification_report(y_true_cv, y_pred_labels)
print(report)
conf_matrix = confusion_matrix(y_true_cv, y_pred_labels)
conf_matrix4 = draw_confusionMatrix(conf_matrix)
conf_matrix4.savefig('conf_matrix4.png')

'''test csv output'''
print('----------use best mlp to predict test csv-------------')
train = pd.read_csv(train_dir, index_col='Id')
train = train.fillna(0)
test = pd.read_csv(test_dir, index_col='Id')
test = test.fillna(0)

sample_submission = pd.read_csv(sample_dir)
y = train["Class"]
X = train.drop(columns='Class')

# standard scale and one hot coding
preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(),
     make_column_selector(dtype_include=object)),
)

X = preprocessor.fit_transform(X)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=11)
test_ = preprocessor.transform(test)

mlp3 = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[57]),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

mlp3.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
history1 = mlp3.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=150, batch_size=32,callbacks=[early_stopping],verbose=0)
print('Done!')

predictions = mlp3.predict(test_)
proba_0 = predictions[:, 0]
proba_1 = 1 - proba_0


sample_submission['Id'] = test.reset_index()['Id']
sample_submission.class_0 = proba_1
sample_submission.class_1 = proba_0
sample_submission.set_index('Id').to_csv('submission.csv')
print(sample_submission)