#!/usr/bin/env python3
import cv2
import pyvirtualcam
from collections import namedtuple
import numpy as np
from multiprocessing import Process, Pipe
from time import time

# NOTE make it a dataclass?
Rect = namedtuple("Rect", ["x", "y", "w", "h"])
Dim = namedtuple("Dim", ["h", "w"])


def make_center_rect(in_dim, out_dim):
    out_aspect = out_dim.h / out_dim.w

    h = in_dim.h
    w = in_dim.w

    center_x = w // 2
    center_y = h // 2

    new_w = int(h / out_aspect)
    new_h = int(w * out_aspect)

    if new_w < w:
        w = new_w
    elif new_h < h:
        h = new_h

    x = center_x - w // 2
    y = center_y - h // 2

    return Rect(x=x, y=y, w=w, h=h)


def main(
    cam_index=0,
    out_dim=Dim(256, 1024),
    speed=0.1,
    demo=False,
    offset_ratio=0.3,
    out_fps=30,
    idle_period=1,
):

    # TODO rewrite to support rectangle output
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
        center_rect = make_center_rect(in_dim, out_dim)
        cur_rect = center_rect
        while True:
            frame_num += 1
            frame = get_frame()

            # face detection
            if parent_conn.poll():
                face = parent_conn.recv()
                if face:
                    target = face
                    last_face_update = time()
                parent_conn.send(frame)

            # if there's no face for a long time, zoom out
            if time() - last_face_update > idle_period:
                target = center_rect

            # smooth step towards target
            cur_rect = np.asarray(cur_rect)
            target = np.asarray(target)
            cur_rect = cur_rect + (target - cur_rect) * speed
            cur_rect = Rect(*(int(i) for i in cur_rect))

            # zoom in
            cur_rect = center_rect  # TODO remove
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
    # TODO respect aspect ratio in case rect collides with border (pad or smth)
    # or should this logic be where we make the Rect()
    # resize
    frame = cv2.resize(
        cropped_frame, (out_dim.w, out_dim.h), interpolation=cv2.INTER_AREA
    )

    return frame


def find_face_worker(conn, offset_ratio):
    # NOTE returns rect face, random aspect ratio
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
        # TODO remove false xd
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
    main(cam_index=0, demo=True)
