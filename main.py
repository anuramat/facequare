#!/usr/bin/env python3
import cv2
import pyvirtualcam
from collections import namedtuple
import numpy as np
from multiprocessing import Process, Pipe

Face = namedtuple("Face", ["x", "y", "w", "h"])


def main(
    cam_index=0,
    out_dim=512,
    speed=0.1,
    demo=False,
    offset_ratio=0.3,
):

    # TODO rewrite to support rectangle output
    """
    cam_index: int -- index of your webcam, usually 0 (the only one in the system)
    detection_period: int -- number of frames between face detection launches
    out_dim: int -- size of virtual webcam output
    speed: float -- zoom speed
    demo: bool -- toggles the opencv output window
    offset_ratio
    """
    # some more init
    frame_num = 0
    last_face_update = float("-inf")
    face = None
    child_conn, parent_conn = Pipe()
    p = Process(target = find_face_worker, args=(child_conn,))
    p.start()
    with (
        pyvirtualcam.Camera(width=out_dim, height=out_dim, fps=30) as out_cam,
        webcam_context(0) as in_cam,
    ):
        while True:
            ret, frame = in_cam.read()
            # face detection
            if frame_num == 0:
                parent_conn.send(frame)
            if parent_conn.poll():
                new_target = parent_conn.recv()
                if new_target:
                    target = new_target
                    last_face_update = frame_num
                parent_conn.send(frame)
            if frame_num - last_face_update > 30 * 3:
                # if there's no face for a long time, show center square
                face = None
            elif face is not None:
                # if we were not focused on the center in the previous frame, 
                # start moving toward the target
                face = np.asarray(face)
                target = np.asarray(target)
                face = face + (target - face) * speed
                face = Face(*(int(i) for i in face))
            else:
                # if we were focused on center, move to the face immediately
                # TODO change: account for offset in the face_find,
                # make center focus a proper face object
                face = target

            frame = face_zoom(frame, face, out_dim, offset_ratio)
            frame_num += 1

            # send frame to virtual cam
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_cam.send(rgb_frame)
            out_cam.sleep_until_next_frame()

            if demo:
                # show gui window, exit on escape key
                cv2.imshow("Input", frame)
                if cv2.waitKey(10) == 27:
                    break


def face_zoom(frame, face, out_dim, offset_ratio):
    if face:
        # make it a square
        side = max(face.h, face.w)
        # pad a little
        offset = int(side * offset_ratio)
        # do the crop
        cropped_frame = frame[
            face.y - offset : face.y + side + offset,
            face.x - offset : face.x + side + offset,
            :,
        ]
    if not face or cropped_frame.size == 0:
        center = (frame.shape[0] // 2, frame.shape[1] // 2)  # sic!
        side = min(frame.shape[0:2])
        x = center[1] - side // 2
        y = center[0] - side // 2
        cropped_frame = frame[y : y + side, x : x + side, :]

    # resize
    frame = cv2.resize(cropped_frame, (out_dim,) * 2, interpolation=cv2.INTER_AREA)

    return frame


def find_face_worker(conn):
    face_cascade_name = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(face_cascade_name)
    while True:
        frame = conn.recv()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        faces = face_cascade.detectMultiScale(frame_gray)
        faces = [Face(*i) for i in faces]
        face = None
        if faces:
            face = max(faces, key=lambda x: x.w * x.h)
        conn.send(face) 

class webcam_context:
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


if __name__ == "__main__":
    main(cam_index=0, demo=True)
