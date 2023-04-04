import cv2
import numpy as np
from triangulation import find_depth_from_disparities
import mediapipe as mp
from dataclasses import dataclass

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


@dataclass
class FaceFeatures:
    eye1: np.array
    eye2: np.array
    mouth: np.array
    nose: np.array


class FeaturesExtractor:
    def __init__(self):
        self.pose = mp_pose.Pose(model_complexity=1,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    def extract_keypts(self, frame):
        frame_height, frame_width, _ = frame.shape

        frame.flags.writeable = False  # TODO: does this improve performance?
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.pose.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame.flags.writeable = True

        if not results.pose_landmarks:
            return (False, None)

        rel_coords = np.array(list(map(lambda pt: (pt.x, pt.y),
                                   results.pose_landmarks.landmark[:11])))
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
            img_coords[2, :], img_coords[5, :], mouth, None)
        return (True, features, img_coords, rel_coords)


#  class StereoDepthEstimator:
#      def __init__(self, cams_separation, focal_length,
#                   depth_to_pixels_size):
#
#          self.pose_left = mp_pose.Pose(model_complexity=1,
#                                        min_detection_confidence=0.5,
#                                        min_tracking_confidence=0.5)
#          self.pose_right = mp_pose.Pose(model_complexity=1,
#                                         min_detection_confidence=0.5,
#                                         min_tracking_confidence=0.5)
#          self.cams_separation = cams_separation
#          self.focal_length = focal_length
#          self.depth_from_disparities = find_depth_from_disparities
#          self.depth_to_pixels_size = depth_to_pixels_size
#
#      def extract_keypts(self, mppose, frame):
#          frame_height, frame_width, _ = frame.shape
#
#          frame.flags.writeable = False  # TODO: does this improve performance?
#          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#          results = mppose.process(frame)
#
#          frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#          frame.flags.writeable = True
#
#          if not results.pose_landmarks:
#              return (False, None)
#
#          coords = np.array(list(map(lambda pt: (pt.x, pt.y),
#                                     results.pose_landmarks.landmark[:11])))
#          #  coords = (coords*np.array((frame_width, frame_height))).astype(int)
#          coords = (coords*np.array((frame_width, frame_height)))
#          #  mp_drawing.draw_landmarks(
#          #      frame,
#          #      results.pose_landmarks,
#          #      mp_pose.POSE_CONNECTIONS,
#          #      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#          #  )
#          return (True, coords)
#
#      def predict(self, frame_left, frame_right):
#
#          ret_left, coords_left = self.extract_keypts(self.pose_left,
#                                                      frame_left)
#          ret_right, coords_right = self.extract_keypts(self.pose_right,
#                                                        frame_right)
#
#          if not ret_left or not ret_right:
#              return (False,  0)
#
#          # ignore the ear keypots
#          coords_right_depth = np.array(list(coords_right[:7]) +
#                                        list(coords_right[9:]))
#
#          coords_left_depth = np.array(list(coords_left[:7]) +
#                                       list(coords_left[9:]))
#
#          depth = self.depth_from_disparities(coords_right_depth[:, 0],
#                                              coords_left_depth[:, 0],
#                                              self.cams_separation,
#                                              self.focal_length
#                                              )
#          return (True, depth, coords_left, coords_right)
