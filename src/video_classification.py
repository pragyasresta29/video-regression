import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Bidirectional, LSTM, BatchNormalization, Reshape, \
    GRU, TimeDistributed, GlobalMaxPool2D, Conv3D, MaxPooling3D
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

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

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=['Light', 'Sedentary', 'MVPA'])


def conv_model(shape):
    model = Sequential()

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
    model.add(Dense(len(label_processor.get_vocabulary()), activation='softmax'))
    return model


def get_model(input_shape):
    model = conv_model(input_shape)
    opt = Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.summary()
    return model


def classification(train, test, split=0.2):
    data, data_labels = train[0], train[1]
    test_data, test_labels = random_test_data(test[0], test[1], int(len(data_labels) * 0.3))
    input_shape = data.shape[1:]

    labels = label_processor(data_labels).numpy()
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # fetch model
    model = get_model(input_shape)
    history = model.fit(data, labels,
                        epochs=15,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stop])

    print("--------------------------------------------")
    y_pred = model.predict(test_data)
    y_pred_bool = np.argmax(y_pred, axis=1)
    report = classification_report(test_labels, y_pred_bool, target_names=['Light', 'Sedentary', 'MVPA'],
                                   output_dict=True)
    print("Test Accuracy: ", report['accuracy'])
    print("Classification Report: \n", classification_report(test_labels, y_pred_bool,
                                                             target_names=['Light', 'Sedentary', 'MVPA']))

    return model


def train_test_split(data, data_labels, split=0.2):
    choice = np.random.choice(len(data_labels), int(len(data_labels) * split), replace=False)
    ind = np.zeros(len(data_labels), dtype=bool)
    ind[choice] = True
    rest = ~ind
    X_train = data[rest]
    X_test = data[ind]
    y_train = data[rest]
    y_test = data[ind]
    return X_train, X_test, y_train, y_test


def random_test_data(data, data_labels, size=300):
    choice = np.random.choice(len(data_labels), size, replace=False)
    ind = np.zeros(len(data_labels), dtype=bool)
    ind[choice] = True
    X_test = data[ind]
    y_test = np.array(data_labels)[ind]
    return X_test, y_test






