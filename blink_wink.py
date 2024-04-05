from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio (eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y) coordinates
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])

    # compue the horizontal distance between the horizontal 
    # eye landmark (x, y) coordinates
    C = dist.euclidean(eye[0],eye[3])

    #compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    #return the eye aspect ratio
    return ear

# the argument parse and parse the argments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help = "path to facial landmark")
#if u want live video stream simply omit the below line when executing the script
ap.add_argument('-v', "--video", type = str, default = "", help = "path to input video file")
args = vars(ap.parse_args()) 


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH_BLINK = 0.18
EYE_AR_THRESH_WINK = 0.23
EYE_AR_CONSEC_FRAMES = 3.5

# initialize the frames counters and the total number of blinks
COUNTER_BLINK = 0
COUNTER_WINK = 0
TOTAL_BLINK = 0
TOTAL_WINK = 0

# initialize dlib's face detector (HOG-based) and create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor..." )
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#grab the indices of the facial landmarks for the left and
# right eye repectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

'''
If youre using a file video stream, then leave the code as is.

Otherwise, if you want to use a built-in webcam or USB camera, uncomment Line 62.

For a Raspberry Pi camera module, uncomment Line 63.

If you have uncommented either Line 62 or Line 63, then uncomment Line 64 as well to indicate that you are not reading a video file from disk.
'''



while True:
    if fileStream and not vs.more():
        break
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart : lEnd]
        rightEye = shape[rStart : rEnd]
        leftEye = shape[lStart : lEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        #compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH_BLINK:
            COUNTER_BLINK += 1
        elif (ear < EYE_AR_THRESH_WINK and ear > EYE_AR_THRESH_BLINK):
            COUNTER_WINK += 1 

        else:
            if COUNTER_BLINK >= EYE_AR_CONSEC_FRAMES:
                TOTAL_BLINK += 1
            
            elif COUNTER_WINK >= EYE_AR_CONSEC_FRAMES:
                TOTAL_WINK += 1

            
            COUNTER_BLINK = 0
            COUNTER_WINK = 0
            
        # draw the total number of blinks on the frame along with the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(TOTAL_BLINK), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)
        cv2.putText(frame, "Winks: {}".format(TOTAL_WINK), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)

    #show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if (key == ord("q")):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


