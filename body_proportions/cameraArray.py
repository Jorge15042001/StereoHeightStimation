
# import the opencv library
import cv2
from time import sleep

import threading
from multiprocessing import Queue 
#  import queue
class cameraThread:

    def __init__(self, camidx,fps = 30):
        self.cam_idx = camidx
        #  self.frames = Queue(fps)
        self.frame = None
        self.closed = False
        self.cap = cv2.VideoCapture(camidx)
        def capture_function ():
            for i in range(30):
                self.cap.grab()
            while self.cap.isOpened() and not self.closed :
                sleep(1/fps)
                frame = self.cap.read()
                #  self.frames.put(frame)
                self.frame = frame
            self.cap.release()
        self.cap_thread = threading.Thread(target=capture_function,daemon=True)
    def close(self):
        self.closed = True
        self.cap_thread.join()

    def get_frame(self):
        return self.frame
        #  return self.frames.get()
        

class CamArray:
    def __init__(self,camidxs,fps = 60):
        self.cams = list(map(lambda x:cameraThread(x,fps), camidxs))
        self.fps = fps

    def start(self):
        for cam in self.cams:
            cam.cap_thread.start()
        while any(map(lambda cam: type(cam.get_frame())==type(None), self.cams)):
            sleep(1/self.fps)
    def close(self):
        for cam in self.cams:
            cam.close()

    def get_frames(self):
        while any(map(lambda cam: not cam.get_frame()[0], self.cams)):continue
        return list(map(lambda cam: cam.get_frame(), self.cams))
    def isOpened(self):
        return all(map(lambda cam: cam.cap.isOpened(), self.cams))



