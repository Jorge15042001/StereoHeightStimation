from datetime import datetime
import cv2
import numpy as np
from calibration import getStereoRectifier
from triangulation import find_depth_from_disparities
import mediapipe as mp
from cameraArray import CamArray

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

B = 16  # Distance between the cameras [cm]
def depth_to_pixels_size(x): return 0.001315666666666666*x+0.00979


left_cam_idx = 2
right_cam_idx = 4
cams = CamArray([left_cam_idx, right_cam_idx])
cams.start()


class StereoHeightStimattor:
    def __init__(self, cams_separation, focal_length,
                 calib_file, depth_to_pixels_size):

        self.pose_left = mp_pose.Pose(model_complexity=1,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
        self.pose_right = mp_pose.Pose(model_complexity=1,
                                       min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5)
        self.cams_separation = cams_separation
        self.focal_length = focal_length
        self.rectify = getStereoRectifier(calib_file)
        self.depth_from_disparities = find_depth_from_disparities
        self.depth_to_pixels_size = depth_to_pixels_size

    def extract_keypts(self, mppose, frame):
        frame_height, frame_width, _ = frame.shape

        frame.flags.writeable = False  # TODO: does this improve performance?
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = mppose.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame.flags.writeable = True

        if not results.pose_landmarks:
            return (False, None)

        coords = np.array(list(map(lambda pt: (pt.x, pt.y),
                                   results.pose_landmarks.landmark[:11])))
        #  coords = (coords*np.array((frame_width, frame_height))).astype(int)
        coords = (coords*np.array((frame_width, frame_height)))
        #  mp_drawing.draw_landmarks(
        #      frame,
        #      results.pose_landmarks,
        #      mp_pose.POSE_CONNECTIONS,
        #      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        #  )
        return (True, coords)

    def predict(self, frame_left, frame_right):
        frame_right, frame_left = self.rectify(frame_right, frame_left)

        ret_left, coords_left = self.extract_keypts(self.pose_left,
                                                    frame_left)
        ret_right, coords_right = self.extract_keypts(self.pose_right,
                                                      frame_right)

        if not ret_left or not ret_right:
            return (False, 0, 0)

        # ignore the ear keypots
        coords_right_depth = np.array(list(coords_right[:7]) +
                                      list(coords_right[9:]))

        coords_left_depth = np.array(list(coords_left[:7]) +
                                     list(coords_left[9:]))

        depth = self.depth_from_disparities(coords_right_depth[:, 0],
                                                coords_left_depth[:, 0],
                                                self.cams_separation,
                                                self.focal_length
                                                )

        eye = np.mean([coords_left[2, :], coords_left[5, :]], axis=0)
        mouth = np.mean([coords_left[9, :], coords_left[10, :]], axis=0)

        mouth_eye = ((eye[0]-mouth[0])**2+(eye[1]-mouth[1])**2)**0.5

        px_size = self.depth_to_pixels_size(depth)
        head_size = mouth_eye * px_size * 3
        height = head_size * 8

        return (True, height, depth)


def putHeightResult(frame_left, frame_right, success_height, height, depth):

    if success_height:
        cv2.putText(frame_right, "Depth: " + str(round(depth, 1)) +
                    " Height:" + str(round(height, 1)), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(frame_left,  "Depth: " + str(round(depth, 1)) +
                    " Height:" + str(round(height, 1)), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    else:
        cv2.putText(frame_right, "right TRACKING LOST", (75, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_left, "left TRACKING LOST", (75, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


if __name__ == "__main__":
    height_stimator = StereoHeightStimattor(B, 829.4, "./stereoMap.xml",
                                            depth_to_pixels_size)

    while cams.isOpened():
        frame_left, frame_right = cams.get_frames()
        succes_left, frame_left = frame_left
        succes_right, frame_right = frame_right

        if not succes_right or not succes_left:
            print("Ignoring empty camera frame.")
            continue

        height_result = height_stimator.predict(frame_left, frame_right)

        putHeightResult(frame_left, frame_right, *height_result)

        cv2.imshow("frame right", frame_right)
        cv2.imshow("frame left", frame_left)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cams.close()
