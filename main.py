#!/usr/bin/env python3
import cv2
import time
import pyvirtualcam
from collections import namedtuple

# TODO move webcam and cascades' init somewhere
# open webcam
cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
# init cascades
face_cascade_name = cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier()
face_cascade.load(face_cascade_name)

def main(demo=True):
    
    # everything is in frames
    frame_num = 0
    last_face_update = -1000
    detection_period = 10 

    out_dim = namedtuple('out_dim', ['width', 'height'])(512, 512)

    face = None
    # TODO rewrite ugly try/except as a context manager for webcam INPUT and maybe cv window?
    try:
        with pyvirtualcam.Camera(width=out_dim.width,height=out_dim.height,fps=30) as cam:
            while True:

                ret, frame = cap.read()

                # face detection
                # TODO run in a separate process
                if frame_num % detection_period == 0:
                    faces = find_faces(frame, face_cascade)
                    # TODO replace faces[0] with biggest square, check that it's not empty
                    if len(faces) != 0:
                        face = faces[0]
                        last_face_update = frame_num
               
                # if there is no face, show as much as possible of the square
                if frame_num - last_face_update > 30*3:
                    face = None

                frame = face_zoom(frame, face, out_dim)

                frame_num += 1
                
                # send frame to virtual cam
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam.send(rgb_frame)
                cam.sleep_until_next_frame()

                if demo:
                    # show gui window, exit on escape key
                    cv2.imshow('Input', frame)
                    if cv2.waitKey(10) == 27:
                        break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def face_zoom(frame, face, out_dim, offset_ratio = 0.3):
    # TODO rewrite to support other shapes maybe?
    # out_dim is (h,w)
    if face is None:
        center = (frame.shape[0]//2, frame.shape[1]//2) # y, x
        side = min(frame.shape[0:2])
        face = (center[1]-side//2, center[0]-side//2,side,side)
        offset_ratio = 0

    x,y,w,h = face
    # make it a square
    side = max(h,w)
    offset = int(side*offset_ratio)
    # TODO check for out of bounds maybe?
    frame = frame[y-offset:y+side+offset, x-offset:x+side+offset, :]

    # resize
    frame = cv2.resize(frame, out_dim, interpolation = cv2.INTER_AREA)  

    return frame

def find_faces(frame, face_cascade):
    # returns [(x,y,w,h)]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    return faces

if __name__ == '__main__':
    main()
