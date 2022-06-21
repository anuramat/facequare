#!/usr/bin/env python3
import cv2
import pyvirtualcam
from collections import namedtuple
import numpy as np
from multiprocessing import Process, Pipe
from time import time

# XXX make it a dataclass?
Rect = namedtuple("Rect", ["x", "y", "w", "h"])
Dim = namedtuple("Dim", ["h", "w"])


def main(
    cam_index=0,
    out_dim=Dim(512, 512),
    speed=0.1,
    demo=False,
    offset_ratio=0.3,
    out_fps=30,
    idle_period=2,
):

    frame_num = 0
    last_face_update = float("-inf")
    child_conn, parent_conn = Pipe()
    p = Process(target=find_face_worker, args=(child_conn, offset_ratio))
    p.start()
    with (
        pyvirtualcam.Camera(width=out_dim.w, height=out_dim.h, fps=out_fps) as out_cam,
        webcam_input(cam_index) as get_frame,
    ):
        frame = get_frame()
        in_dim = Dim(h=frame.shape[0], w=frame.shape[1])
        parent_conn.send(frame)
        center_rect = adapt_rect(in_dim, out_dim, Rect(0, 0, in_dim.w, in_dim.h))
        cur_rect = center_rect
        while True:
            frame_num += 1
            frame = get_frame()

            # face detection
            if parent_conn.poll():
                face = parent_conn.recv()
                if face:
                    target = adapt_rect(in_dim, out_dim, face)
                    last_face_update = time()
                parent_conn.send(frame)

            # if there's no face for a long time, zoom out
            if time() - last_face_update > idle_period:
                target = center_rect

            # smooth step towards target
            # TODO doesn't exactly respect the aspect ratio
            cur_rect = np.asarray(cur_rect)
            target = np.asarray(target)
            cur_rect = cur_rect + (target - cur_rect) * speed
            cur_rect = Rect(*(int(i) for i in cur_rect))

            # zoom in
            frame = rect_zoom(frame, cur_rect, out_dim)

            # send frame to virtual cam
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_cam.send(rgb_frame)
            out_cam.sleep_until_next_frame()

            # show gui window, exit on escape key
            if demo:
                cv2.imshow("Input", frame)
                if cv2.waitKey(10) == 27:
                    break


def rect_zoom(frame, rect, out_dim):
    # crop
    cropped_frame = frame[
        rect.y : rect.y + rect.h,
        rect.x : rect.x + rect.w,
        :,
    ]
    # resize
    frame = cv2.resize(
        cropped_frame, (out_dim.w, out_dim.h), interpolation=cv2.INTER_AREA
    )

    return frame


def find_face_worker(conn, offset_ratio):
    face_cascade_name = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(face_cascade_name)
    while True:
        frame = conn.recv()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        faces = face_cascade.detectMultiScale(frame_gray)
        faces = [Rect(*i) for i in faces]
        face = None
        if faces:
            face = max(faces, key=lambda x: x.w * x.h)
            offset = int(offset_ratio * (face.h + face.w) / 2)
            face = Rect(
                face.x - offset,
                face.y - offset,
                face.w + offset * 2,
                face.h + offset * 2,
            )
        conn.send(face)


# XXX make a Rect method?
def adapt_rect(in_dim, out_dim, rect):

    w = rect.w
    h = rect.h
    center_x = rect.x + w // 2
    center_y = rect.y + h // 2
    rect_aspect = h / w

    out_aspect = out_dim.h / out_dim.w

    # 1) calculate new_rect.h,w (should fit in in_dim) (cropped image should fully contain rect)
    # 1a) make an out_dim shaped rectangle, containing "rect"
    # by changing either h or w so that it matches aspect ratio
    if out_aspect > rect_aspect:
        h = int(w * out_aspect)
    elif out_aspect < rect_aspect:
        w = int(h / out_aspect)

    # 1b) make this out_dim shaped rectangle fit in in_dim shaped rectangle
    # by changing h/w proportionally
    if w > in_dim.w:
        h = int(h * in_dim.w / w)
        w = in_dim.w
    if h > in_dim.h:
        w = int(w * in_dim.h / h)
        h = in_dim.h

    # 2) if any of the borders (x, x+w, y, y+h) are out of bounds of [0:h,0:w] => subtract difference from x/y
    x = center_x - w // 2
    y = center_y - h // 2

    if x < 0:
        x = 0
    elif x + w > in_dim.w:
        x = in_dim.h - h

    if y < 0:
        y = 0
    elif y + h > in_dim.h:
        y = in_dim.h - h

    result = Rect(x, y, w, h)

    return result


class webcam_input:
    def __init__(self, cam_index):
        self.cam_index = cam_index

    def __enter__(self):
        # open webcam
        self.cap = cv2.VideoCapture(self.cam_index)
        # Check if the webcam is opened correctly
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        return lambda: self.cap.read()[1]

    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
