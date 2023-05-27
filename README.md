# Estimation Energy Cost from Video using 3D-CNN
Deep learning model to predict mets(metabolic equivalent) from video. 

# Setup Instructions
1. Install Python version 3.8
2. The libraries used in this project can be found in req.txt. To install this using pip: `pip install -r req.txt`

# Using Scripts
We will use the scripts in the following order for preprocessing and running our experiment
1. **Step 1:** `src/video_preprocessing.py` - This script is used to convert each video data into 1 minute intervals. We need to run this for each video for now but can be modified to take a list of videos. Before running the script, you need to update the following: 
  a. `input_file_path` - Path of the input Video
  b. `output_dir_path` - Path to store the output video intervals
  c. `video_start` - Date and time of the where the video starts. This is done so that we can same the date and time of each minute interval in it's filename so that we dont have to annotate it manually. 
  d. `datetime_str` - Usually set this to date & time we want to start our interval. For eg: for Video1, the annotated date consists of intervals every 26th second of the video.
  e. `subject` - subject id 
_**Note**: We need to run this once for each video we have. _

2. **Step 2:** `src/map-data.py` - This script creates a csv file by synchronizing the 1 minute-video intervals with it's respective METS value and category from the annotated data. We need to update the following values: 
  a. `input_file_path` - Path of the input Video
  b. `output_dir_path` - Path to store the output video intervals
  c. `data_path` - Path where we will storing our mapped data
  d. `datetime_str` - Starting time of the first video for the subject. 
  e. `subject` - subject id 
_**Note**: We need to run this once for each video/annotated csv file we have. _

3. **Step 3:** `src/preprocessing.py` - This script is used to reesize and fetch desired no. of frames from 1 minute video intervals. The `subject`, `video_dir_path`, & `data_path` need to updated for each subjects video data. The frames dimensions and no. of frames are set to the ones that I used for my research. 

4. **Step 3:** `src/experiment.py` - This script is used to perform both classification and regression using the specified train and test data. We need to update the input directory and data path and set the train and test subject to run this script. The classification and regression model are in separate scripts. The model parameters can be modified in these scripts before running the experiment. 







