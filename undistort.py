import pickle
import cv2
import numpy as np


# Read in the saved objpoints and imgpoints
cameraMatrix, dist = pickle.load(open( "E:\Documents\MachineLearningUDEMY\Vision\CameraCalibration\calibration.pkl", "rb" ))
#dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )

# Read in an image
img = cv2.imread('E:\Documents\MachineLearningUDEMY\Vision\CameraCalibration\cali5.png')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

# Undistort
undistorted = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
undistorted = undistorted[y:y+h, x:x+w]
undistorted = cv2.resize(undistorted,(640,480))
cv2.imshow("Original Image", img)

cv2.imshow("Undistorted Image", undistorted)

cv2.waitKey(0)
# Destroy all the windows
cv2.destroyAllWindows()
