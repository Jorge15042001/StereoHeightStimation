import cv2
import numpy as np
import calibration
import triangulation as tri
import mediapipe as mp
from time import sleep
from cameraArray import CamArray
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

from datetime import datetime
B = 16               #Distance between the cameras [cm]
depth_to_pixels_size = lambda x: 0.001315666666666666*x+0.00979
frame_count = 0
person_id = 0
#  output_file = open("./dataset/output.txt","a")

def filp_correction (img):
    return img [::-1][:,::-1]

class RollingMean():
    def __init__(self,cap:int):
        self.values = []
        self.cap = cap
    def get(self)->float:
        #  return np.median(self.values)
        return sum(self.values) /len(self.values)
    def register(self,val:float)->float:
        self.values.append(val)
        if len(self.values)  == self.cap:
            self.values.pop(0)
        return self.get()
pose_left = mp_pose.Pose(model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)
pose_right= mp_pose.Pose(model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)

left_cam_idx = 2
right_cam_idx = 4
cams = CamArray([left_cam_idx,right_cam_idx])
cams.start()

height_mean = RollingMean(2)

while(cams.isOpened()):
  frame_count += 1
  capture_dateime = datetime.now()
  cap_time_str = capture_dateime.strftime("%Y-%m-%d--%H-%M-%f") 

  frame1,frame2 = cams.get_frames()
  succes_left,frame_left= frame1
  succes_right,frame_right= frame2
  #  frame_left= filp_correction(frame_left)
  #  frame_right= filp_correction(frame_right)

  if not succes_right or not succes_left:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    continue
  #  if frame_count%60 ==0:
  #      cv2.imwrite(f"./dataset/input/{cap_time_str}_{person_id}_left.png",frame_right)
  #      cv2.imwrite(f"./dataset/input/{cap_time_str}_{person_id}_right.png",frame_left)
  ################## CALIBRATION #########################################################

  frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)
  #  if frame_count%60 ==0:
  #      cv2.imwrite(f"./dataset/input/{cap_time_str}_{person_id}_left_rectify.png",frame_right)
  #      cv2.imwrite(f"./dataset/input/{cap_time_str}_{person_id}_right_rectify.png",frame_left)

  ########################################################################################
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  frame_right.flags.writeable = False
  frame_left.flags.writeable = False

  # Convert the BGR image to RGB
  frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
  frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

  # Process the image and find faces
  results_right = pose_right.process(frame_right)
  results_left = pose_left.process(frame_left)

  # Convert the RGB image to BGR
  frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
  frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)

  h,w, _= frame_right.shape

  # Draw the pose annotation on the image.
  frame_right.flags.writeable = True
  frame_left.flags.writeable = True


  mp_drawing.draw_landmarks(
      frame_right,
      results_right.pose_landmarks,
      mp_pose.POSE_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

  mp_drawing.draw_landmarks(
      frame_left,
      results_left.pose_landmarks,
      mp_pose.POSE_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
  ################## CALCULATING DEPTH #########################################################

  if not results_right.pose_landmarks or not results_left.pose_landmarks:
      cv2.putText(frame_right, "right TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
      cv2.putText(frame_left, "left TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
  else:
      coords_right = np.array(list(map(lambda pt: (pt.x,pt.y),results_right.pose_landmarks.landmark[:11])))
      coords_right = (coords_right*np.array([w,h])).astype(int)
      coords_right_depth = np.array(list(coords_right[:7])+list(coords_right[9:]))
      

      coords_left = np.array(list(map(lambda pt: (pt.x,pt.y),results_left.pose_landmarks.landmark[:11])))
      coords_left = (coords_left*np.array([w,h])).astype(int)
      coords_left_depth = np.array(list(coords_left[:7])+list(coords_left[9:]))
      

      #  for corx,cory in coords_left:
      #      frame_left[cory:cory+10,corx:corx+10] = 0
      #  for corx,cory in coords_right:
      #      frame_right[cory:cory+10,corx:corx+10] = 0

      depth = abs(tri.find_depth_from_disparities(coords_right_depth[:,0],coords_left_depth[:,0], B))

      eye = np.mean([coords_left[2,:],coords_left[5,:]],axis=0)
      mouth = np.mean([coords_left[9,:],coords_left[10,:]],axis=0)
      #  frame_left[eye[1]:eye[1]+10,eye[0]:eye[0]+10] = 0
      #  frame_left[mouth[1]:mouth[1]+10,mouth[0]:mouth[0]+10] = 0
      mouth_eye= ((eye[0]-mouth[0])**2+(eye[1]-mouth[1])**2)**0.5

      
      px_size = depth_to_pixels_size(depth)
      head_size = mouth_eye * px_size *3 
      height = head_size *8
      #  height = height_mean.register(height)
      cv2.putText(frame_right, "Depth: " + str(round(depth,1))+ " Height:" +str(round(height,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),2)
      cv2.putText(frame_left,  "Depth: "+ str(round(depth,1)) + " Height:" +str(round(height,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),2)
      #  output_file.write(f"{cap_time_str},{depth},{height}\n")

  
  # Flip the image horizontally for a selfie-view display.

  #  cv2.imshow("frame right", cv2.resize(frame_right,(960,720)))
  #  cv2.imshow("frame left", cv2.resize(frame_left,(960,720)))

  #  if frame_count%60 ==0:
      #  cv2.imwrite(f"./dataset/output/{cap_time_str}_{person_id}_right.png", frame_right)
      #  cv2.imwrite(f"./dataset/output/{cap_time_str}_{person_id}_left.png", frame_left)

  cv2.imshow("frame right", frame_right)
  cv2.imshow("frame left", frame_left)
  if cv2.waitKey(5) & 0xFF == 27:
    break

cams.close()
