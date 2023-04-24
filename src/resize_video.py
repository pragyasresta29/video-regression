import cv2

input_file_path = '/Users/pragya/Downloads/ProjectVideo/CO1002_2018-03-29 07-26-06.mp4'


ORG_DIM = (1280, 720)
DIM = 124
# computing new width & height to maintain aspect ratio
r = DIM / ORG_DIM[0]
dim = (DIM, int(ORG_DIM[1] * r))

print("Resized Image dimensions: ", dim)

# Define a video capture object
vidcap = cv2.VideoCapture(input_file_path)

# Capture video frame by frame
success, image = vidcap.read()

# Declare the variable with value 0
count = 0

# Creating a loop for running the video
# and saving all the frames
while success:

    # Capture video frame by frame
    success, image = vidcap.read()

    # Resize the image frames
    resize = cv2.resize(image, dim)

    # Saving the frames with certain names
    cv2.imwrite("/Users/pragya/Downloads/ProjectVideo/%04d.jpg" % count, resize)
    break
    # Closing the video by Escape button
    if cv2.waitKey(10) == 27:
        break

    # Incrementing the variable value by 1
    count += 1

