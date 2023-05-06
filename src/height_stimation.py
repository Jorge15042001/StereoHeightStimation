import cv2
import numpy as np
from calibration import getStereoRectifier
from triangulation import find_depth_from_disparities
import mediapipe as mp
from cameraArray import CamArray
from featuresExtractor import FaceFeatures, FeaturesExtractor
from utils import startCameraArray, loadStereoCameraConfig, StereoConfig

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


def computeHeigth2(features: FaceFeatures, pixel_size: float, cam_center):
    mid_eye_y = np.mean((features.eye1, features.eye2), axis=0)[1]
    cam_center_y = cam_center[1]

    height = (mid_eye_y-cam_center_y)*pixel_size * -1

    #  height = head_size * 8
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
    stereo_config = loadStereoCameraConfig("./stereo_config.json")
    #  cam_center_left, cam_center_right = cam_centers
    f_length = min(stereo_config.left_camera.fpx,
                   stereo_config.right_camera.fpx)
    cams = startCameraArray(stereo_config.left_camera,
                            stereo_config.right_camera)
    cams.start()

    rectify = getStereoRectifier("./stereoMap.xml")
    features_left_extractor = FeaturesExtractor()
    features_right_extractor = FeaturesExtractor()

    while cams.isOpened():
        frame_left, frame_right = cams.get_frames()
        succes_left, frame_left = frame_left
        succes_right, frame_right = frame_right

        if not succes_right or not succes_left:
            print("Ignoring empty camera frame.")
            continue

        frame_left, frame_right = rectify(frame_left, frame_right)

        features_left = features_left_extractor.extract_keypts(frame_left)
        features_right = features_right_extractor.extract_keypts(frame_right)

        if not features_left[0] or not features_right[0]:
            continue

        depth = computeDepth(features_left[2], features_right[2],
                             stereo_config.cam_separation, f_length)

        px_size = stereo_config.depth_to_pixel_size * depth
        #  height = computeHeigth(features_left[1], px_size)
        height = computeHeigth2(
            features_left[1], px_size, stereo_config.left_camera.center)

        putHeightResult(frame_left, frame_right, True, height, depth)

        cv2.imshow("frame right", cv2.resize(
            frame_right, (np.array(frame_right.shape[:2][::-1])*1.5).astype(int)))
        cv2.imshow("frame left", cv2.resize(
            frame_left, (np.array(frame_right.shape[:2][::-1])*1.5).astype(int)))
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cams.close()
