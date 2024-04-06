from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import keyboard
import morse_code
import constants

# Based the blinking detection off of this tutorial:
# https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib
# by Adrian Rosebrock from pyimagesearch.

# dlib pre-trained facial landmark predictor available at
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Also seems to be available @
# https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2

# HELLO WORLD = .... . .-.. .-.. --- / .-- --- .-. .-.. -..


def main():
    # Parse predictor argument
    arg_par = argparse.ArgumentParser()
    arg_par.add_argument("-p", "--shape-predictor", required=True,
                         help="path to facial landmark predictor")
    arg_par.add_argument("-v", "--video", required=True,
                         help="path to input video file")
    args = vars(arg_par.parse_args())

    (vs, detector, predictor, lStart,
     lEnd, rStart, rEnd) = setup_detector_video(args)
    total_morse = loop_video(vs, detector, predictor, lStart,
                             lEnd, rStart, rEnd)
    cleanup(vs)
    print_results(total_morse)

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    eye_ar = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return eye_ar

def setup_detector_video(args):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # initialize the video stream
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["video"])

    return vs, detector, predictor, lStart, lEnd, rStart, rEnd


def loop_video(vs, detector, predictor, lStart, lEnd, rStart, rEnd):
    COUNTER = 0
    BREAK_COUNTER = 0
    EYES_OPEN_COUNTER = 0
    CLOSED_EYES = False
    WORD_PAUSE = False
    PAUSED = False

    total_morse = ""
    morse_word = ""
    morse_char = ""

    # loop over frames from the video file stream
    while True:
        ret, frame = vs.read()

        if not ret:
            break

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            left_eye_ar = eye_aspect_ratio(leftEye)
            right_eye_ar = eye_aspect_ratio(rightEye)
            eye_ar = (left_eye_ar + right_eye_ar) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if eye_ar < constants.EYE_AR_THRESH:
                COUNTER += 1
                BREAK_COUNTER += 1
                if COUNTER >= constants.EYE_AR_CONSEC_FRAMES:
                    CLOSED_EYES = True
                if not PAUSED:
                    morse_char = ""
                if (BREAK_COUNTER >= constants.BREAK_LOOP_FRAMES):
                    break
            else:
                if (BREAK_COUNTER < constants.BREAK_LOOP_FRAMES):
                    BREAK_COUNTER = 0
                EYES_OPEN_COUNTER += 1
                if COUNTER >= constants.EYE_AR_CONSEC_FRAMES_CLOSED:
                    morse_word += "-"
                    total_morse += "-"
                    morse_char += "-"
                    COUNTER = 0
                    CLOSED_EYES = False
                    PAUSED = True
                    EYES_OPEN_COUNTER = 0
                elif CLOSED_EYES:
                    morse_word += "."
                    total_morse += "."
                    morse_char += "."
                    COUNTER = 1
                    CLOSED_EYES = False
                    PAUSED = True
                    EYES_OPEN_COUNTER = 0
                elif PAUSED and (EYES_OPEN_COUNTER >= constants.PAUSE_CONSEC_FRAMES):
                    morse_word += "/"
                    total_morse += "/"
                    morse_char = "/"
                    PAUSED = False
                    WORD_PAUSE = True
                    CLOSED_EYES = False
                    EYES_OPEN_COUNTER = 0
                    keyboard.write(morse_code.from_morse(morse_word))
                    morse_word = ""
                elif (WORD_PAUSE and EYES_OPEN_COUNTER >= constants.WORD_PAUSE_CONSEC_FRAMES):
                    total_morse += "¦/"
                    morse_char = ""
                    WORD_PAUSE = False
                    CLOSED_EYES = False
                    EYES_OPEN_COUNTER = 0
                    keyboard.write(morse_code.from_morse("¦/"))

            cv2.putText(frame, "EAR: {:.2f}".format(eye_ar), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "{}".format(morse_char), (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            print("\033[K", "morse_word: {}".format(morse_word), end="\r")

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("]") or (BREAK_COUNTER >= constants.BREAK_LOOP_FRAMES):
            keyboard.write(morse_code.from_morse(morse_word))
            break

    return total_morse


def cleanup(vs):
    cv2.destroyAllWindows()
    vs.release()


def print_results(total_morse):
    print("Morse Code: ", total_morse.replace("¦", " "))
    print("Translated: ", morse_code.from_morse(total_morse))


if __name__ == "__main__":
    main()