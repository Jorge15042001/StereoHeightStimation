import cv2
import numpy as np
from triangulation import find_depth_from_disparities
import mediapipe as mp
from dataclasses import dataclass
from enum import Enum

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh


@dataclass
class FaceFeatures:
    eye1: np.array
    eye2: np.array
    mouth: np.array
    nose: np.array


class FeaturesExtractor:
    def __init__(self):
        #  self.pose = mp_pose.Pose(model_complexity=1,
        #                           min_detection_confidence=0.5,
        #                           min_tracking_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    #  def extract_keypts(self, frame):
    #      frame_height, frame_width, _ = frame.shape
    #
    #      frame.flags.writeable = False  # TODO: does this improve performance?
    #      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #
    #      results = self.pose.process(frame)
    #
    #      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #      frame.flags.writeable = True
    #
    #      if not results.pose_landmarks:
    #          return (False, None)
    #
    #      rel_coords = np.array(list(map(lambda pt: (pt.x, pt.y),
    #                                 results.pose_landmarks.landmark[:11])))
    #      #  coords = (coords*np.array((frame_width, frame_height))).astype(int)
    #      img_coords = (rel_coords*np.array((frame_width, frame_height)))
    #      #  mp_drawing.draw_landmarks(
    #      #      frame,
    #      #      results.pose_landmarks,
    #      #      mp_pose.POSE_CONNECTIONS,
    #      #      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    #      #  )
    #      mouth = np.mean([img_coords[9, :], img_coords[10, :]], axis=0)
    #
    #      features = FaceFeatures(
    #          img_coords[2, :], img_coords[5, :], mouth, None)
    #      return (True, features, img_coords, rel_coords)

    def extract_keypts(self, frame):
        frame_height, frame_width, _ = frame.shape

        frame.flags.writeable = False  # TODO: does this improve performance?
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame.flags.writeable = True

        if not results.multi_face_landmarks:
            return (False, None)

        rel_coords = np.array(list(map(lambda pt: (pt.x, pt.y),
                                   results.multi_face_landmarks[0].landmark)))
        #  coords = (coords*np.array((frame_width, frame_height))).astype(int)
        img_coords = (rel_coords*np.array((frame_width, frame_height)))
        #  mp_drawing.draw_landmarks(
        #      frame,
        #      results.pose_landmarks,
        #      mp_pose.POSE_CONNECTIONS,
        #      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        #  )
        mouth = np.mean([img_coords[9, :], img_coords[10, :]], axis=0)

        features = FaceFeatures(
            img_coords[33, :], img_coords[263, :], None, None)
        return (True, features, img_coords, rel_coords)
