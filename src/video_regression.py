import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Bidirectional, LSTM, BatchNormalization, Reshape, \
    GRU, TimeDistributed, GlobalMaxPool2D, Conv3D, MaxPooling3D
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_validate
from sklearn.utils import class_weight

from tensorflow_addons.metrics import RSquare
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings("ignore")

BATCH_SIZE = 32
EPOCHS = 10

MAX_SEQ_LENGTH = 60
NUM_FEATURES = 2048
MAX_FRAMES = 60

ORG_DIM = (1280, 720)
DIM = 80
# computing new width & height to maintain aspect ratio
r = DIM / ORG_DIM[0]
dim = (DIM, int(ORG_DIM[1] * r))

#Save Processed data to pickle file
processed_data_path = '/Users/pragya/PycharmProjects/NLP/video-regression/src/data/processed/'


def rsquare(y_true, y_pred):
    SS_res = keras.backend.sum(keras.backend.square(y_true - y_pred))
    SS_tot = keras.backend.sum(keras.backend.square(y_true - keras.backend.mean(y_true)))
    return 1 - SS_res / (SS_tot + keras.backend.epsilon())


def conv_model(shape):
    model = Sequential()
    print(shape)
    # 3D convolutional layers
    model.add(Conv3D(32, kernel_size=(2, 2, 2), activation='relu', input_shape=shape))
    model.add(MaxPooling3D())
    model.add(BatchNormalization())

    model.add(Conv3D(64, kernel_size=(2, 2, 2), activation='relu'))
    model.add(MaxPooling3D())
    model.add(BatchNormalization())

    model.add(Conv3D(128, kernel_size=(2, 2, 2), activation='relu'))
    model.add(MaxPooling3D())
    model.add(BatchNormalization())

    # Flatten the output of the convolutional layers
    model.add(Flatten())

    # Dense layers for regression
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    return model


def get_model(input_shape):
    model = conv_model(input_shape)
    print(input_shape)
    opt = Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError(), rsquare])
    model.summary()
    return model


def regression(train, test):
    data, data_labels = train[0], train[1]
    test_data, test_labels = test[0], test[1]
    input_shape = data.shape[1:]
    print("Input Shape: ", input_shape)
    data_labels = np.array(data_labels)

    early_stop = EarlyStopping(monitor='val_root_mean_squared_error', patience=5, verbose=1)

    model = get_model(input_shape)
    history = model.fit(data, data_labels,
                       epochs=25,
                       batch_size=32,
                       validation_split=0.2,
                       callbacks=[early_stop])

    # testing
    y_pred = model.predict(test_data)
    rmse = mean_squared_error(test_labels, y_pred, squared=False)
    print("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse))

    r2 = r2_score(test_labels, y_pred)
    print("The R2 score on test set: {:.4f}".format(r2))
    return model


def train_test_split(data, data_labels, split=0.2):
    choice = np.random.choice(len(data_labels), int(len(data_labels) * split), replace=False)
    ind = np.zeros(len(data_labels), dtype=bool)
    ind[choice] = True
    rest = ~ind
    X_train = data[rest]
    X_test = data[ind]
    y_train = data_labels[rest]
    y_test = data_labels[ind]
    return X_train, X_test, y_train, y_test


def random_test_data(data, data_labels, size=300):
    choice = np.random.choice(len(data_labels), size, replace=False)
    ind = np.zeros(len(data_labels), dtype=bool)
    ind[choice] = True
    X_test = data[ind]
    y_test = np.array(data_labels)[ind]
    return X_test, y_test