import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import cv2
    import numpy as np
    import certifi
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import pickle
    import matplotlib.pyplot as plt

    return mo, np, pickle, plt


@app.cell
def _(pickle):
    with open('dump.pkl', 'rb') as f:
        dump = pickle.load(f)
    return (dump,)


@app.cell
def _(mo, np, plt):
    class Snap:
        def __init__(self, obj):
            self.obj = obj
        
        def _display_(self):
           return mo.hstack([
               plt.imshow(self.obj['img']), 
               mo.vstack([mo.md('face landmarks shape'), np.array(self.obj['result'].face_landmarks).shape])])
           #return mo.inspect(self.obj['result'])
    return (Snap,)


@app.cell
def _(Snap, dump):
    [Snap(_dv) for _dv in dump.values()]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
