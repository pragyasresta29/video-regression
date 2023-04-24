import pandas as pd
import numpy as np

import cv2
import os
import pickle


# Resize and fetch desired no. of frames from 1 minute video intervals

# Define video parameters

MAX_SEQ_LENGTH = 60  # max no. of sequence of frames to return
FRAMES_JUMP = 3  # get 1 frame every (frame_jump) seconds in the video

ORG_DIM = (1280, 720)
DIM = 80
# computing new width & height to maintain aspect ratio
r = DIM / ORG_DIM[0]
dim = (DIM, int(ORG_DIM[1] * r))

print("Resized Image dimensions: ", dim)


subject = '1004'
# Set the path of the input video file
video_dir_path = '/Users/pragya/Downloads/ProjectVideo/video_data/' + subject + '/'
data_path = '/Users/pragya/PycharmProjects/NLP/video-regression/src/data/' + subject + '_data.csv'


def load_video(path, max_frames=MAX_SEQ_LENGTH, resize=dim):
    cap = cv2.VideoCapture(path)
    frames = []
    assert cap.isOpened()

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_out = 1

    index_in = -1
    index_out = -1
    try:
        while True:
            success = cap.grab()
            if not success: break
            index_in += 1
            """
            # convert fps of video to 1 fps to get 1 key frame each second 
            # All the videos are 1 minute. 
            """

            out_due = int(index_in / fps_in * fps_out)
            if out_due > index_out:
                success, frame = cap.retrieve()
                if not success: break
                index_out += 1
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)
                if len(frames) == max_frames:
                    break
    finally:
        cap.release()
    return np.array(frames[::FRAMES_JUMP])


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["filename"].values.tolist()
    labels = (df["mets_standard"].values, df['mets_category'].values)

    frame_features = np.empty(
        shape=(num_samples, int(MAX_SEQ_LENGTH/FRAMES_JUMP), dim[1], dim[0], 3), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        print(idx)
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]
        frame_features[idx] = frames

    return frame_features, labels


df = pd.read_csv(data_path)
data, data_labels = prepare_all_videos(df, video_dir_path)


#Save Processed data to pickle file
processed_data_path = '/Users/pragya/PycharmProjects/NLP/video-regression/src/data/processed/'

with open(processed_data_path + subject+ '_data_labels_' + str(DIM) + '.pkl', 'wb') as f:
    pickle.dump(data_labels, f)

with open(processed_data_path + subject + '_data_' + str(DIM) + '.pkl', 'wb') as f:
    pickle.dump(data, f)




