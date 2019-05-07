import numpy as np
import math
from glob import glob
#import promaci_extras
import os
import cv2


dir = '/home/jmateo/Documents/USB abril 13/2018_04_10/Kalibrierung'



# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)


#Blob detector has to be used here for improved accuracy of points detection
########################################Blob Detector##############################################

# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 0
blobParams.maxThreshold = 80

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 10     # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 700  # maxArea may be adjusted to suit for your experiment

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1

# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.4

# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(blobParams)

###################################################################################################

###################################################################################################


#array of points 4x11 means 4 rows always, and 5 columns and 6 columns for asymmetric grids
objp = np.zeros((11*4,3), np.float32)
objp[:, :2] = np.mgrid[0:4, 0:11].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#change extension depending on type of images
images = sorted(glob(os.path.join(dir, '*.tiff')))

cont = 0
for fname in images:
    cont +=1
    print (cont)
    img = cv2.imread(fname)
    height, width = img.shape[:2]
    img = cv2.resize(img, (width//3, height//3))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)




    # Find the chess board corners
    # ret, corners = cv2.findChessboardCorners(gray, (7,4), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)
    # ret, corners = cv2.findCirclesGrid(gray, (5, 5), cv2.CALIB_CB_ADAPTIVE_THRESH)

    ret, corners = cv2.findCirclesGrid(gray, (4, 11), None, cv2.CALIB_CB_ASYMMETRIC_GRID, detector)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (4, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (4, 11), corners2, ret)
        #cv2.namedWindow("img", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
        cv2.imshow('img', img)
        cv2.waitKey()
        #cv2.imwrite(dir + '/processed/calib' + '{0:04}'.format(cont) +'.png', img)

img = cv2.imread(images[2])
img = cv2.resize(img, (width // 3, height // 3))
h, w = img.shape[:2]
# return the camera matrix, distortion coefficients, rotation and translation vectors

#perform the camera calibration
ret, cam_mtx, distortion_coeff , rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
#ret, cam_mtx, distortion_coeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#show calculated parameters
# print ('camera matrix:' + str(cam_mtx))
# print ('distortion coff: ' + str(distortion_coeff))
# print ('Rotation: ' + str(rvecs))
# print ('Translation: '+ str(tvecs))

newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cam_mtx,distortion_coeff,(w,h),1,(w,h))


# undistort method 1
dst = cv2.undistort(img, cam_mtx, distortion_coeff, None, newcameramtx)

cv2.imshow('undistorted image', dst)
cv2.namedWindow('undistorted image', cv2.WINDOW_NORMAL)
cv2.waitKey()
# crop the image
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]


# undistort method 2
mapx,mapy = cv2.initUndistortRectifyMap(cam_mtx, distortion_coeff, None, newcameramtx, (w,h), 5)
dst2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# crop the image
x,y,w,h = roi
dst3 = dst2[y:y+h, x:x+w]



cv2.imshow('undistorted image 2', dst2)
cv2.namedWindow('undistorted image 2', cv2.WINDOW_NORMAL)
cv2.imshow('original image', img)
cv2.waitKey()

#calculate error and show it
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cam_mtx, distortion_coeff)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error += error

print("total error: ", total_error/len(objpoints))


#write processed images if required
#cv2.imwrite(dir + '/processed/' + 'correction' + '.png', dst2)
#cv2.imwrite(dir + '/processed/' + 'correction_crop' + '.png', dst3)