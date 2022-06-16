#!/usr/bin/env python3
import cv2
import time
import pyvirtualcam
from collections import namedtuple
import numpy as np

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

def main(demo=False):
    
    # everything is in frames
    frame_num = 0
    last_face_update = -1000
    detection_period = 5

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
                    # TODO replace faces[0] with biggest square, check that it's not empty and sides are about 1:1
                    if len(faces) != 0:
                        target = faces[0]
                        last_face_update = frame_num
                
                # if there is no face, show center square
                if frame_num - last_face_update > 30*3:
                    face = None
                elif face is not None:
                    # smoothing
                    face = np.asarray(face)
                    target = np.asarray(target)
                    face = face + (target - face) * 0.1
                    face = tuple(int(i) for i in face)
                else:
                    face = target

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

    new_frame = frame[y-offset:y+side+offset, x-offset:x+side+offset, :]
    # TODO rewrite, might infinitely recurse if you mess up
    if new_frame.shape[0] == 0 or new_frame.shape[1] == 0:
        return face_zoom(frame, None, out_dim, offset_ratio)

    # resize
    new_frame = cv2.resize(new_frame, out_dim, interpolation = cv2.INTER_AREA)  

    return new_frame

def find_faces(frame, face_cascade):
    # returns [(x,y,w,h)]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    return faces

if __name__ == '__main__':
    main(demo=True)
