import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import cv2 as cv
    import numpy as np
    import matplotlib.pyplot as plt

    return cv, mo, np, plt


@app.cell
def _(cv, mo):
    _haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    mo.stop(_haar_cascade.empty())
    return


@app.cell
def _(cv, plt):
    cap= cv.VideoCapture(1)
    ret, _frame = cap.read()
    if not ret:
        print("Frame grab failed")
    f = _frame
    plt.imshow(f)
    return (f,)


@app.cell
def _(f, np, plt):
    import os
    import urllib.request
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    MODEL_PATH = "face_landmarker.task"
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        output_facial_transformation_matrixes=True,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE,
    )

    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    vis = rgb.copy()
    h, w = vis.shape[:2]
    normal = None

    if result.face_landmarks:
        lm = result.face_landmarks[0]
        pts = np.array([(p.x * w, p.y * h, p.z * w) for p in lm])  # 478 pts, incl. iris

        if result.facial_transformation_matrixes:
            R = np.array(result.facial_transformation_matrixes[0])[:3, :3]
            normal = R[:, 2] / np.linalg.norm(R[:, 2])  # canonical model's z-axis -> camera space

        RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_IRIS = [468, 469, 470, 471, 472]  # 468 = pupil center
        LEFT_IRIS = [473, 474, 475, 476, 477]   # 473 = pupil center

        for idx in RIGHT_EYE + LEFT_EYE:
            cv2.circle(vis, tuple(pts[idx][:2].astype(int)), 1, (0, 255, 0), -1)
        for idx in RIGHT_IRIS + LEFT_IRIS:
            cv2.circle(vis, tuple(pts[idx][:2].astype(int)), 2, (255, 0, 0), -1)
        for c in (468, 473):
            cv2.circle(vis, tuple(pts[c][:2].astype(int)), 3, (255, 255, 0), -1)

        if normal is not None:
            nose = pts[1][:2]
            end = nose + normal[:2] * 100
            cv2.arrowedLine(vis, tuple(nose.astype(int)), tuple(end.astype(int)), (0, 0, 255), 2)
            print(f"normal = {normal}")
    else:
        print("no face detected")

    plt.figure(figsize=(7, 7))
    plt.imshow(vis)
    plt.axis("off")
    plt.title("green=eye contour  blue=iris  yellow=pupil center  red=normal")
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
