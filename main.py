"""
Live face mesh + iris tracking with head-pose normal vector.

Controls:
    q / ESC   quit
    n         toggle normal vector arrow
    e         toggle eye/iris points
    a         add point + image to tracking data 

Requires:
    pip install opencv-python mediapipe numpy certifi
"""

import os
import ssl
import time
import urllib.request

import cv2
import numpy as np
import certifi
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_IRIS = [468, 469, 470, 471, 472]  # 468 = pupil center
LEFT_IRIS = [473, 474, 475, 476, 477]   # 473 = pupil center

CAMERA_INDEX = 1
WINDOW_NAME = "face tracker (q/ESC quit, n normal, e eyes)"


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading face_landmarker.task ...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(MODEL_URL, context=ctx) as resp, open(MODEL_PATH, "wb") as out:
        out.write(resp.read())
    print("Done.")


def build_landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        output_facial_transformation_matrixes=True,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.FaceLandmarker.create_from_options(options)


def process_frame(landmarker, frame_bgr, timestamp_ms, show_normal, show_eyes):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]

    if result.face_landmarks:
        lm = result.face_landmarks[0]
        pts = np.array([(p.x * w, p.y * h, p.z * w) for p in lm])

        if show_eyes:
            for idx in RIGHT_EYE + LEFT_EYE:
                cv2.circle(vis, tuple(pts[idx][:2].astype(int)), 1, (0, 255, 0), -1)
            for idx in RIGHT_IRIS + LEFT_IRIS:
                cv2.circle(vis, tuple(pts[idx][:2].astype(int)), 2, (255, 0, 0), -1)
            for c in (468, 473):
                cv2.circle(vis, tuple(pts[c][:2].astype(int)), 3, (255, 255, 0), -1)

        if show_normal and result.facial_transformation_matrixes:
            R = np.array(result.facial_transformation_matrixes[0])[:3, :3]
            normal = R[:, 2] / np.linalg.norm(R[:, 2])
            nose = pts[1][:2]
            end = nose + normal[:2] * 100
            cv2.arrowedLine(vis, tuple(nose.astype(int)), tuple(end.astype(int)), (0, 0, 255), 2)
    else:
        cv2.putText(vis, "no face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    return vis, result


def main():
    tosave = {}
    ensure_model()
    landmarker = build_landmarker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    show_normal = True
    show_eyes = True
    start_time = time.time()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            timestamp_ms = int((time.time() - start_time) * 1000)
            vis, res = process_frame(landmarker, frame_bgr, timestamp_ms, show_normal, show_eyes)
            cv2.imshow(WINDOW_NAME, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            elif key == ord("n"):
                show_normal = not show_normal
            elif key == ord("e"):
                show_eyes = not show_eyes
            elif key == ord('a'):
                tosave[timestamp_ms] = {
                   'result': res,
                   'img': frame_bgr.copy()
                }
                print('saved frame')

            # also handle the window's [x] close button
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        if (keys_to_save := len(tosave.keys())) > 0:
            with open('dump.pkl', 'wb') as file:
                pickle.dump(tosave, file)
                print(f'dumped {keys_to_save} frames.')

if __name__ == "__main__":
    main()
