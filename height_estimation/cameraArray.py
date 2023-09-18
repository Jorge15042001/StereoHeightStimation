
# import the opencv library
import cv2
from time import sleep
from datetime import datetime
from typing import Callable, Tuple

import threading
import numpy as np


def get_now_str():
    return datetime.utcnow().strftime('%Y-%m-%d__%H:%M:%S.%f')


class cameraThread:
    """
    Handles access to a single camera

    Args:
        camidx (int): camera descriptor
        fps (float): frames per second
        reslution (tuple[int, int]): camera capture resolution

    Attributes:
        camidx (int): camera descriptor
        frame (np.array): current frame
        close (bool): controls capture thread main loop
        cap (cv2.VideoCapture): capture object
        cap_thread (threading.Thread): capture thread
    """

    def __init__(self, camidx: int, fps: float = 30,
                 resolution: Tuple[int, int] = (640, 480)):

        self.cam_idx: int = camidx
        self.frame: Tuple[bool, np.ndarray[np.uint8]] | None = None
        self.closed: bool = False
        self.cap: cv2.VideoCapture = cv2.VideoCapture(camidx)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # capture thread function
        def capture_function() -> None:
            # Synchronization: grab a lot frames quickly so the cameras
            for i in range(30):
                self.cap.grab()
            # Main thread
            while self.cap.isOpened() and not self.closed:
                # read frames faster than the camera's fsp
                sleep(1/(fps*3))
                frame = self.cap.read()
                self.frame = frame
            self.cap.release()
        # Capture thread
        self.cap_thread: threading.Thread = threading.Thread(
            target=capture_function, daemon=True)

    def close(self) -> None:
        """ Stop capturing frames """
        self.closed = True
        self.cap_thread.join()

    def get_frame(self) -> np.ndarray[np.uint8]:
        """ Get latest captured frame

        Returns:
            frame (np.array): the latest captured frame
        """
        return self.frame


class CamArray:
    """
    Handles access to a stereo imaging system

    Args:
        camidxs (list[int]): camera descriptors
        fps (float): frames per second
        reslution (tuple[int, int]): camera capture resolution
        save_frames_to (str | None): path to folder where to save the frames, if None then no fames are saved
        rectifier:  Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] function that rectifies the stereo image pair


    Attributes:
        cams(list[cameraThread]): cameraThread Objects for each camera
        fps (float): capture fps
        save_frames_to (str | None): path to folder where the frames will be saved
        video_writters (list[cv2.VideoWriter_fourcc] | None): frames writers for capture frames
        video_writters_rectified (list[cv2.VideoWriter_fourcc] | None): frames writers for rectified frames
        rectifier (Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]): function that rectifies the stereo image pair
    """

    def __init__(self,
                 camidxs: list[int],
                 fps: float = 60,
                 save_frames_to: str | None = None,
                 rectifier: Callable[[np.ndarray, np.ndarray],
                                     Tuple[np.ndarray, np.ndarray]] =
                 lambda x, y: (x, y),
                 resolution: Tuple[int, int] = (640, 480)):

        assert len(camidxs) == 2, "Only 2 camera array suported"
        self.cams: list[cameraThread] = [cameraThread(x, fps) for x in camidxs]
        self.fps: float = fps
        self.save_frames_to: str | None = save_frames_to

        self.video_writers: list[cv2.VideoWriterr_fourcc] | None = None
        self.video_writers_rectified: list[cv2.VideoWriterr_fourcc] | None = None

        if save_frames_to is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            now_str = get_now_str()

            self.video_writers = [cv2.VideoWriter(
                f"{save_frames_to}/{now_str}_{cam_id}.avi",
                fourcc, fps, resolution) for cam_id in camidxs]

            self.video_writers_rectified = [cv2.VideoWriter(
                f"{save_frames_to}/{now_str}_{cam_id}_rectified.avi",
                fourcc, fps, resolution) for cam_id in camidxs]

        self.rectifier: Callable[[np.ndarray, np.ndarray],
                                 Tuple[np.ndarray, np.ndarray]] = rectifier

    def start(self) -> None:
        """ Starts capturing images from the camera array """
        for cam in self.cams:
            cam.cap_thread.start()
        # Wait for the all the cameras to capture at least one frame
        while any(map(lambda cam: type(cam.get_frame()) == type(None), self.cams)):
            sleep(1/self.fps)

    def close(self) -> None:
        """ Stops capturing images from the camera array """
        for cam in self.cams:
            cam.close()

    def get_frames(self) -> list[np.ndarray[np.uint8]]:
        """ Get the latests frame from each camera in the array"""
        # wait for all the cameras to have atleast one frame available
        while any(map(lambda cam: not cam.get_frame()[0], self.cams)): continue
        frames = [cam.get_frame() for cam in self.cams]
        #  frames = list(map(lambda cam: cam.get_frame(), self.cams))
        # record the frames into a video
        if self.video_writers is not None:
            for (_, frame), writer in zip(frames, self.video_writers):
                writer.write(frame)

        frame1, frame2 = self.rectifier(frames[0][1], frames[1][1])
        frames = ((True, frame1), (True, frame2))

        if self.video_writers_rectified is not None:
            for (_, frame), writer in zip(frames, self.video_writers_rectified):
                writer.write(frame)
        return ((True, frame1), (True, frame2))


    def isOpened(self):
        """ Check if the camera array is opened"""
        return all(map(lambda cam: cam.cap.isOpened(), self.cams))
