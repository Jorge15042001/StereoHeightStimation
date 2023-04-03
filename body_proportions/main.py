# import the opencv library
import cv2

from cameraArray import CamArray

# define a video capture object
left_cam_idx = 2
right_cam_idx = 4
cams = CamArray([left_cam_idx, right_cam_idx])
cams.start()


def filp_correction(img):
    return img[::-1][:, ::-1]


while(True):

    frame1, frame2 = cams.get_frames()
    _, frame1 = frame1
    _, frame2 = frame2
    frame1 = filp_correction(frame1)

    #  Display the resulting frame
    cv2.imshow('frame1', frame1)
    cv2.imshow('frame2', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
