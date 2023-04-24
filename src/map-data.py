import pandas as pd
import datetime
import warnings

warnings.filterwarnings("ignore")

subject = '1004'
input_file_path = '/Users/pragya/PycharmProjects/NLP/thesis/AIM1_MERGED_NOLDUS_CHAMBER_RMR/CO_' + subject + '_1_NOLDUS_CHAMBER_RMR.csv'
video_dir_path = '/Users/pragya/Downloads/ProjectVideo/video_data/' + subject
datetime_str = '03/19/18 06:56:58'  # Starting time of the first video.

data_path = '/Users/pragya/PycharmProjects/NLP/video-regression/src/data/'

time_obj = datetime.datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')

print(input_file_path, video_dir_path)


# mapping video data and energy metrics data
def map_data():
    df = pd.read_csv(input_file_path, low_memory=False)

    # Mapping Posture Code based on posture_noldus
    df['Posture'] = df['posture_noldus'].apply(lambda posture: map_posture(posture))
    df['new_time'] = pd.to_datetime(df['date'] + " " + df['time'])

    temp_df = df[(~df['mets_standard'].isna()) & (df['new_time'] > time_obj)]
    temp_df['code'] = temp_df['new_time'].apply(lambda time: find_max_postures(df, time))
    temp_df['mets_category'] = temp_df.apply(lambda row: map_mets_category(row['mets_standard'], row['code']), axis=1)

    temp_df['new_time'] = temp_df['new_time'].dt.strftime("%Y-%m-%d %H:%M:%S")
    temp_df['filename'] = temp_df['subject'].astype(str) + "_" + temp_df["new_time"].astype(str) + '.mp4'

    new_df = temp_df[['filename', 'mets_standard', 'mets_category']]
    csv_name = data_path + subject + "_data.csv"
    new_df.to_csv(csv_name, index=False)


def map_mets_category(mets, code):
    if mets < 1.5 or (mets < 2.0 and code == 'Posture1'):
        return 'Sedentary'  #
    if code == 'Posture2' or (mets >= 1.5 and mets < 3.0):
        return 'Light'  # Light Activity
    return 'MVPA'  # Anything greater or equal to 3.0 is considered moderated to vigorous physical activity


def map_posture(posture):
    if posture in ["sitting", "lying", "crouching/kneeling/squatting"]:
        return "Posture1"
    if posture == "standing":
        return "Posture2"

    # includes ['intermittent movement', 'other - posture', 'cycling','walking', 'dark/obscured/oof', 'stepping']
    return "Posture3"


def find_max_postures(df, time):
    start_time = time - pd.Timedelta(minutes=1)
    return df[(df['new_time'] >= start_time) & (df['new_time'] < time)]['Posture'].mode()[0]


map_data()