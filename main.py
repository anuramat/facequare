#!/usr/bin/env python3
import cv2
import pyvirtualcam
from collections import namedtuple
import numpy as np

Face = namedtuple('Face', ['x', 'y', 'w', 'h'])

def main(cam_index = 0, detection_period = 5, out_dim = 512, speed=0.1, demo=False, offset_ratio=0.3):
    #TODO rewrite to support rectangle output
    '''
    cam_index: int -- index of your webcam, usually 0 (the only one in the system)
    detection_period: int -- number of frames between face detection launches
    out_dim: int -- size of virtual webcam output
    speed: float -- zoom speed
    demo: bool -- toggles the opencv output window
    offset_ratio
    '''
    # init cascades
    face_cascade_name = cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(face_cascade_name)
    # some more init 
    frame_num = 0
    last_face_update = float('inf')
    face = None
    with (pyvirtualcam.Camera(width=out_dim,height=out_dim,fps=30) as out_cam,
            webcamContext(0) as in_cam):
        while True:
            ret, frame = in_cam.read()

            # face detection
            # TODO run in a separate process
            if frame_num % detection_period == 0:
                faces = find_faces(frame, face_cascade)
                if faces:
                    target = max(faces, key = lambda x: x.w * x.h)
                    last_face_update = frame_num
            
            # if there is no face, show center square
            if frame_num - last_face_update > 30*3:
                face = None
            elif face is not None:
                # smoothing
                face = np.asarray(face)
                target = np.asarray(target)
                face = face + (target - face) * speed
                face = Face(*(int(i) for i in face))
            else:
                face = target

            frame = face_zoom(frame, face, out_dim, offset_ratio)

            frame_num += 1
            
            # send frame to virtual cam
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_cam.send(rgb_frame)
            out_cam.sleep_until_next_frame()

            if demo:
                # show gui window, exit on escape key
                cv2.imshow('Input', frame)
                if cv2.waitKey(10) == 27:
                    break

def face_zoom(frame, face, out_dim, offset_ratio):
    if face:
        # make it a square
        side = max(face.h,face.w)
        # pad a little
        offset = int(side*offset_ratio)
        # do the crop
        cropped_frame = frame[face.y-offset:face.y+side+offset, face.x-offset:face.x+side+offset, :]
    if not face or cropped_frame.size == 0:
        center = (frame.shape[0]//2, frame.shape[1]//2) # sic!
        side = min(frame.shape[0:2])
        x = center[1]-side//2
        y = center[0]-side//2
        cropped_frame = frame[y:y+side, x:x+side, :]

    # resize
    frame = cv2.resize(cropped_frame, (out_dim,)*2, interpolation = cv2.INTER_AREA)  

    return frame


def find_faces(frame, face_cascade):
    # returns [(x,y,w,h)], (x,y) is the center of the face
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    return [Face(*i) for i in faces]

class webcamContext:
    def __init__(self, cam_index):
        self.cam_index = cam_index

    def __enter__(self):
        # open webcam
        self.cap = cv2.VideoCapture(self.cam_index)
        # Check if the webcam is opened correctly
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        return self.cap

    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(cam_index = 0, demo=True)
