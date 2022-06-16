#!/usr/bin/env python3
import cv2
import time
import pyvirtualcam

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

    frame_num = 0

    face = None
    # TODO rewrite ugly try/except as a context manager for webcam INPUT and maybe cv window?
    try:
        with pyvirtualcam.Camera(width=512,height=512,fps=30) as cam:
            while True:

                ret, frame = cap.read()
                # calc initial middle square
                center = (frame.shape[0]//2, frame.shape[1]//2) # y, x
                side = min(frame.shape[0:2])

                # TODO add "or no face for a long time"
                if face is None:
                    face = (center[1]-side//2, center[0]-side//2,side,side)


                # face detection
                # TODO run in a separate process
                if frame_num % 10 == 0:
                    faces = find_faces(frame, face_cascade)
                    # returns [(x,y,w,h)]
                    # TODO replace faces[0] with biggest square
                    if len(faces) != 0:
                        face = faces[0]
                
                # TODO smoothing (square is a target, hardcode speed, move on each iteration)
                frame = face_zoom(frame, face, 512, 512) 

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

def face_rect(frame, face):
    x,y,w,h = face
    top_left_corner = (x, y)
    bottom_right_corner  = (x+w, y+h)
    frame = cv2.rectangle(frame, top_left_corner, bottom_right_corner, (0,255,0), 3)  
    return frame

def face_zoom(frame, face, height, width):
    x,y,w,h = face
    # make it a square I guess
    side = max(w,h)
    offset = side//3
    # TODO check for out of bounds maybe?
    frame = frame[y-offset:y+side+offset, x-offset:x+side+offset, :]

    # resize
    frame = cv2.resize(frame, (height, width), interpolation = cv2.INTER_AREA)  

    return frame

def find_faces(frame, face_cascade):
    # returns [(x,y,w,h)]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    return faces

if __name__ == '__main__':
    main()
