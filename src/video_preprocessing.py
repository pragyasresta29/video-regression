from moviepy.video.io.VideoFileClip import VideoFileClip
import datetime, time
import pandas as pd

# We will be converting each video into 1-min intervals with this script
# Set the path of the input video file
# Need to change this value for every video input
input_file_path = '/Users/pragya/Downloads/ProjectVideo/CO1004_2018-03-19 06-55-36.mp4'
data_path = '/Users/pragya/PycharmProjects/NLP/video-regression/src/data/'
video_start = '03/19/18 06:55:38'
datetime_str = '03/19/18 17:29:58'
subject = '1004'

# Set the path of the output video file directory
output_dir_path = '/Users/pragya/Downloads/ProjectVideo/video_data/' + subject + "/"

# Create a VideoFileClip object for the input video
clip = VideoFileClip(input_file_path)

# Get the duration of the input video in seconds
duration = clip.duration

# Set the duration of each interval in seconds
interval_duration = 60

start = datetime.datetime.strptime(video_start, '%m/%d/%y %H:%M:%S')
time_obj = datetime.datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
newtime=time_obj + datetime.timedelta(minutes=1)

time_diff=time_obj - start
int_start = int(time_diff.total_seconds()) + 1
print(int_start)

print(newtime)

# Loop through each interval and extract the video clip
for i in range(int_start, int(duration), interval_duration):
    # Set the start and end times of the interval
    start_time = i
    end_time = i + interval_duration

    print(start_time, end_time, newtime)
    # # Extract the video clip for the interval
    interval_clip = clip.subclip(start_time, end_time)
    #
    # # Set the output file name for the interval clip - the time is always set to the end of the video interval
    output_file_name = subject + '_' + str(newtime) + '.mp4'
    #
    # # Write the interval clip to the output file
    interval_clip.write_videofile(output_dir_path + output_file_name)
    newtime = newtime + datetime.timedelta(minutes=1)


