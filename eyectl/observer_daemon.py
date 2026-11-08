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

from collections import deque
from pathlib import Path
import threading
import cv2

class FrameBuffer:
  def __init__(self, device, window_s=3.0, width=1200, height=800, fps=10 ):
    self.cap = cv2.VideoCapture(device)
    if not self.cap.isOpened():
      raise RuntimeError(f'cannot open camera device {device}')
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    self.cap.set(cv2.CAP_PROP_FPS, fps)
    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    self.window_ns = int(window_s * 1e9)
    self._buf = deque()
    self._lock = threading.Lock()
    self._running = True
    self._thread = threading.Thread(target=self._loop, daemon=True)
    self._thread.start()
  
  def _loop(self):
    while self._running:
      ok, frame = self.cap.read() # blocks, releases GIL
      t = time.monotonic_ns()
      if not ok:
        time.sleep(0.1)
        continue
      with self._lock:
        self._buf.append((t, frame))
        while self._buf and t - self._buf[0][0] > self.window_ns:
          self._buf.popleft()
  
  def slice(self, t0_ns, t1_ns):
    with self._lock:
      return [(t,f) for t,f in self._buf if t0_ns <= t <= t1_ns] 
  
  def latest(self):
    with self._lock:
      return self._buf[-1] if self._buf else None 
  
  def close(self):
    self.running = False
    self._thread.join(timeout=1)
    self.cap.release()
  
def save_clip(frames, out_dir): 
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  for i, (t,f) in enumerate(frames):
    cv2.imwrite(str(out / f'{i:04d}_{t}.jpg'), f)

if __name__ == '__main__':
  fb = FrameBuffer(1)
  time.sleep(3.5)
  now = time.monotonic_ns()
  clip = fb.slice(now - 2_000_000_000, now)
  if len(clip) > 1:
    span_s = (clip[-1][0] - clip[0][0]) / 1e9
    print(f"{len(clip)} frames over {span_s:.2f}s")
    print(f'frame shape {clip[0][1].shape}')
  save_clip(clip, 'test')
  fb.close()