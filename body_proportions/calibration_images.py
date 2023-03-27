import cv2
from cameraArray import CamArray 
import os

images_dir = "./images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)



left_cam_idx = 2
right_cam_idx = 4
cams = CamArray([left_cam_idx,right_cam_idx])
cams.start()

num = 0

def filp_correction (img):
    return img [::-1][:,::-1]

def save_images(img1,img2,num):
    cv2.imwrite(f'{images_dir}/imageL' + str(num) + '.png', img1)
    cv2.imwrite(f'{images_dir}/imageR' + str(num) + '.png', img2)
    print("images saved!")

def preview_and_save(img1,img2,num):
    while True:
        k2 = cv2.waitKey(5)
        if k2 == ord("q"): break
        if k2 == ord("s"): 
            save_images(img, img2, num)
            num += 1
            break

while cams.isOpened():

    #  succes1, img = cap.read()
    #  succes2, img2 = cap2.read()
    frame1,frame2 = cams.get_frames()
    _,img= frame1
    _,img2= frame2
    img = filp_correction(img)
    img2 = filp_correction(img2)


    k = cv2.waitKey(5)

    if k == 27:
        break
    if k == ord('s'): # preview the image
        preview_and_save(img, img2, num)
        num += 1

    cv2.imshow('Img 1',img)
    cv2.imshow('Img 2',img2)
