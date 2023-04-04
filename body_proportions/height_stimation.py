import cv2
import numpy as np
from calibration import getStereoRectifier
from triangulation import find_depth_from_disparities
import mediapipe as mp
from cameraArray import CamArray
from featuresExtractor import FaceFeatures, FeaturesExtractor
from utils import loadCamArrayFromJson

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def depth_to_pixels_size(x): return 0.001315666666666666*x+0.00979


def computeDepth(keypoinsL, keypoinsR, cams_sep, f_length):
    return find_depth_from_disparities(keypoinsL[:, 0], keypoinsR[:, 0],
                                       cams_sep, f_length)


def computeHeigth(features: FaceFeatures, pixel_size: float):
    mid_eye = np.mean((features.eye1, features.eye2), axis=0)
    mouth_eye = ((mid_eye[0]-features.mouth[0])**2 +
                 (mid_eye[1]-features.mouth[1])**2) ** 0.5
    head_size = mouth_eye * pixel_size * 3
    height = head_size * 8
    return height


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
    B = 16  # Distance between the cameras [cm]
    cams = loadCamArrayFromJson("./stereo_config.json")
    cams.start()
    print("cameras started")

    rectify = getStereoRectifier("./stereoMap.xml")
    features_left_extractor = FeaturesExtractor()
    features_right_extractor = FeaturesExtractor()

    while cams.isOpened():
        frame_left, frame_right = cams.get_frames()
        print(frame_left)
        succes_left, frame_left = frame_left
        succes_right, frame_right = frame_right

        if not succes_right or not succes_left:
            print("Ignoring empty camera frame.")
            continue

        frame_left, frame_right = rectify(frame_left, frame_right)
        features_left = features_left_extractor.extract_keypts(frame_left)
        features_right = features_right_extractor.extract_keypts(frame_right)

        if not features_left[0] or not features_right[0]:
            putHeightResult(frame_left, frame_right, False, 0, 0)
            continue
        depth = computeDepth(features_left[2], features_right[2], B, 829.4)
        px_size = depth_to_pixels_size(depth)
        height = computeHeigth(features_left[1], px_size)

        putHeightResult(frame_left, frame_right, True, height, depth)

        cv2.imshow("frame right", frame_right)
        cv2.imshow("frame left", frame_left)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cams.close()
