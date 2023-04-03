import cv2
from cameraArray import CamArray
from datetime import datetime
import os

images_dir = "./images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)


left_cam_idx = 2
right_cam_idx = 4
cams = CamArray([left_cam_idx, right_cam_idx])
cams.start()

def flip_correction(img):
    return img[::-1][:, ::-1]


def save_images(img1, img2):
    now = datetime.now()
    timestamp = now.strftime("%m.%d.%Y_%H:%M:%S.%f")
    cv2.imwrite(f'{images_dir}/imageL__{timestamp}.png', img1)
    cv2.imwrite(f'{images_dir}/imageR__{timestamp}.png', img2)
    print("images saved!")


def preview_and_save(img1, img2):
    while True:
        k2 = cv2.waitKey(5)
        if k2 == ord("q"):
            break
        if k2 == ord("s"):
            save_images(img1, img2)
            break


def capture_images(cams:CamArray):
    if len(cams.cams) !=2:
        print(f"No support for {len(cams.cams)} cameras, only use 2 cameras ")
        exit(-1)


    while cams.isOpened():
        frame_left, frame_right = cams.get_frames()
        _, img_left = frame_left
        _, img_right = frame_right

        k = cv2.waitKey(5)

        if k == ord('q'):
            break

        if k == ord('s'):  # preview the image
            preview_and_save(img_left, img_right)

        cv2.imshow('Img 1', img_left)
        cv2.imshow('Img 2', img_right)
