import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# from video-classification import *
from video_regression import regression
from video_classification import classification
import numpy as np
import pandas as pd
import time

e_start_time = time.perf_counter()
processed_data_path = '/Users/pragya/PycharmProjects/NLP/video-regression/src/data/processed/'
csv_data_path = '/Users/pragya/PycharmProjects/NLP/video-regression/src/data/'

MAX_SEQ_LENGTH = 60  # max no. of sequence of frames to return
FRAMES_JUMP = 3  # get 1 frame every (frame_jump) seconds in the video

ORG_DIM = (1280, 720)
DIM = 80
# computing new width & height to maintain aspect ratio
r = DIM / ORG_DIM[0]
dim = (DIM, int(ORG_DIM[1] * r))


def get_data(subs=['1001']):
    final_data = None
    reg_labels = []
    class_labels = []
    for subject in subs:
        data_path = processed_data_path + subject + '_data_' + str(DIM) + '.pkl'
        data_label_path = processed_data_path + subject + '_data_labels_' + str(DIM) + '.pkl'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        with open(data_label_path, 'rb') as f:
            data_labels = pickle.load(f)

        if (final_data is None):
            final_data = data
        else:
            final_data = np.concatenate((final_data, data), axis=0)
        reg_labels = [*reg_labels, *data_labels[0]]
        class_labels = [*class_labels, *data_labels[1]]
    return final_data, reg_labels, class_labels


def get_data_statistics(subs=['1001']):
    df = pd.DataFrame()
    for subject in subs:
        csv = csv_data_path + subject + "_data.csv"
        temp = pd.read_csv(csv)
        df = pd.concat([df, temp])
    print('----------------DATA STATISTICS-----------------')
    total = df.shape[0]
    print('Total Data: ', total)
    print('Subjects: ', subs)
    print('----------------Regression Stats----------------')
    print('METs Mean value: ', df['mets_standard'].mean())
    print('METs Min value: ', df['mets_standard'].min())
    print('METs Max value: ', df['mets_standard'].max())
    print('----------------Classification Stats----------------')
    print("Mets Categories: ", df['mets_category'].unique())
    print("Class Distribution: ")
    value_counts = df['mets_category'].value_counts()
    print("[Class]: \t[Count], [%]")
    print("Sedentary: \t%d, \t%f " % (value_counts[0], (value_counts[0] * 100) / total))
    print("Light: \t\t%d, \t%f " % (value_counts[1], (value_counts[1] * 100) / total))
    print("MVPA: \t\t%d, \t%f " % (value_counts[2], (value_counts[2] * 100) / total))
    print('------------------------------------------------')


def evaluate_classification(model, test_data, test_labels):
    y_pred = model.predict(test_data)
    y_pred_bool = np.argmax(y_pred, axis=1)
    print("Classification Report: \n", classification_report(test_labels, y_pred_bool,
                                                             target_names=['Light', 'Sedentary', 'MVPA']))


subjects = ['1001', '1002', '1003', '1004']
get_data_statistics(subjects)

train_subjects = ['1001']
print("Train Subjects: ", train_subjects)
get_data_statistics(train_subjects)
train_data, train_reg_labels, train_class_labels = get_data(train_subjects)

test_subjects = ['1002']
print("Test Subjects: ", test_subjects)
get_data_statistics(test_subjects)
test_data, test_reg_labels, test_class_labels = get_data(test_subjects)

# Perform regression
start_time = time.perf_counter()
r_model = regression((train_data, train_reg_labels), (test_data, test_reg_labels))
end_time = time.perf_counter() - start_time
print("#Regression: Total time: ", end_time)

# Perform classification
start_time = time.perf_counter()
c_model = classification((train_data, train_class_labels), (test_data, test_class_labels))
end_time = time.perf_counter() - start_time
print("# Classification: Total time: ", end_time)

e_end_time = time.perf_counter() - e_start_time
print("Total Time Taken for Experiment: ", e_end_time)

model_name = '_'.join(train_subjects) + '.hdf5'
r_model.save(processed_data_path + '_r_model_' + model_name)
c_model.save(processed_data_path + '_c_model_' + model_name)

