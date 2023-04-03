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
        self.cap = cv2.VideoCapture(camidx)
        def capture_function ():
            for i in range(30):
                self.cap.grab()
            while True:
                sleep(1/fps)
                frame = self.cap.read()
                #  self.frames.put(frame)
                self.frame = frame
                print(f"cam {camidx} captured 1 frame")
             
        self.cap_thread = threading.Thread(target=capture_function,daemon=True)

    def get_frame(self):
        return self.frame
        #  return self.frames.get()
        

class CamArray:
    def __init__(self,camidxs,fps = 60):
        self.cams = list(map(lambda x:cameraThread(x,fps), camidxs))
        self.fps = fps


        ## weird trick to synchronize the cameras
        ## source https://stackoverflow.com/questions/21671139/how-to-synchronize-two-usb-cameras-to-use-them-as-stereo-camera
    def start(self):
        for cam in self.cams:
            cam.cap_thread.start()
        while any(map(lambda cam: type(cam.get_frame())==type(None), self.cams)):
            sleep(1/self.fps)


    def get_frames(self):
        return list(map(lambda cam: cam.get_frame(), self.cams))





# define a video capture object
cams = CamArray([2,4])
cams.start()


def filp_correction (img):
    return img [::-1][:,::-1]

while(True):

    frame1,frame2 = cams.get_frames()
    _,frame1 = frame1
    _,frame2 = frame2
    frame1 = filp_correction(frame1)

    #  Display the resulting frame
    cv2.imshow('frame1', frame1)
    cv2.imshow('frame2', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

