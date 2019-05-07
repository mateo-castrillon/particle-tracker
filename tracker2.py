import cv2
import numpy as np
import math
#from glob import glob
from utils import promaci_extras
from collections import deque # this is to create a buffer so the tail of the path doesn't last forever

import inspect
import os

scriptname = inspect.getframeinfo(inspect.currentframe()).filename
root_dir = os.path.dirname(os.path.abspath(scriptname))


#sorted(glob(os.path.join(silh_dir, '*.png')) + glob(os.path.join(silh_dir, '*.jpg')))
dir = "/home/staff/mo.schulze/python4ia/examples/videos/"

#load video
cap = cv2.VideoCapture(dir + '50 Hz Belichtungszeit 5000-1.avi')
#determine frames per second of loaded video
fps = cap.get(cv2.CAP_PROP_FPS)
#print fps

#check if video is correctly loaded
if (cap.isOpened() == False):
    print("Error opening video file, check file location and name")

#to count frames
cont = 0

#uncomment if required to save data
file = open(root_dir + '/tracking_points.data', 'w')



# Set up the SimpleBlobdetector with default parameters.-------------------
'''
SimpleBlobDetedtor works by doing different thresholds to the iamge (let's say every 10 levels of intensity) between
a min and a max thresh value, with this it is determined which connected regions are present in most of the thresholds done
and therefore are more likely to be our actual particles... they can be then filtered by size or shape with further parameters
'''

params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 100
#params.maxArea = 800

# Filter by Circularity
params.filterByCircularity = False
#params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.2

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.1

detector = cv2.SimpleBlobDetector_create(params)
#-----------------------------------------------------------------------


#Create background frame
ret, background = cap.read()
#select ROI from background so we don't process innecesary regions
bbox = cv2.selectROI(background, False)
cv2.destroyWindow('ROI selector')
#crop image based on selected region
background = own_fns.CropImage(background, bbox)

#create background substractor based on Mixture of Gaussians
fgbg = cv2.createBackgroundSubtractorMOG2()
fgmask = fgbg.apply(background[:,:,1])

#read nex frame and crop it
ret, frame1 = cap.read()
frame1 = own_fns.CropImage(frame1, bbox)

#Aplly Gaussian Blur (filtering) to minimize noise
fgauss = cv2.GaussianBlur(frame1[:, :, 1], (5, 5), 0)

# absolute substraction of filtered frame and background created
bg_diff = cv2.absdiff(fgauss, fgmask)

# threshold the image so that we have a binary image of the background elements
ret, bg = cv2.threshold(bg_diff, 60, 255, cv2.THRESH_BINARY_INV)
#ret_bg, bg = cv2.threshold(bg_diff,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#cv2.imshow('bg', bg)


#lists to store trajectories
center = list()
center_float = list()
keypoints_history = [[]]

center.append(None)
center_float.append(None)

#pts = deque(maxlen=args["buffer"]) #for vanishing line
pts = deque(maxlen=50) # this one can be adjusted to have longer or smaller tail



current_pos = None
last_pos = None
no_frame_cont = 1

#loop through all frames
while(cap.isOpened()):

    # pass current frame of video as RGB image to frame object

    #get current frame
    ret, frame = cap.read()

    #if frame is empty, we are finished
    if frame is None:
        print ('All frames read')
        break


    #crop the frame
    frame = promaci_extras.CropImage(frame, bbox)

    #apply the same procedure done to the background
    fgauss = cv2.GaussianBlur(frame[:, :, 1], (5, 5), 0)
    diff = cv2.absdiff(fgauss, fgmask)
    ret, bin = cv2.threshold(frame[:,:,1], 60, 255, cv2.THRESH_BINARY)
    #ret, bin = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #cv2.imshow('threshold', bin)
    #cv2.imshow('bg', bg)

    #NOW magic part: apply a XOR between the binary image of the background and the binary image of the current frame
    #as a result only the elements visible in either of the images goes to 1 the others to 0, which will be the moving particles
    #bin =cv2.bitwise_xor(bin,bg)

    #morphological operation to filter noise
    #k1 = np.ones((5, 5), np.uint8)
    k2 = np.ones((3, 3), np.uint8)
    bin = cv2.erode(bin, k2,)
    bin = cv2.morphologyEx(bin, cv2.MORPH_OPEN, k2)

    #cv2.imshow('xor + open', bin)
    #get a copy of current frame to draw on it
    drawing = frame.copy()

    #apply the simple blob detector to the resulting image
    keypoints = detector.detect(255-bin)

    #draw resulting keypoints
    im_with_keypoints = cv2.drawKeypoints(drawing, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DEFAULT)


    #save keypoints in our dedicated lists
    #TODO: list of lists to store all particles and have a multiple tracker
    #change index [0] to observe veolcity from other particles.
    if len(keypoints) > 0:
        center.append((int(keypoints[0].pt[0]), int(keypoints[0].pt[1])))
        center_float.append(keypoints[0].pt)


        pts.appendleft(center[cont + 1])


        for i in range(len(keypoints)):

            cv2.putText(im_with_keypoints, 'kp: ' + str(i) + ' ' + str((int(keypoints[i].pt[0]), int(keypoints[i].pt[1]))),
                       (int(keypoints[i].pt[0])+15, int(keypoints[i].pt[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)


    else:
        center_float.append(None)
        center.append(None)
        pts.appendleft(None)


    current_pos = center_float[-1]
    last_pos = center_float[-2]

    #when any of the positions is not a number go to next iterarion
    if current_pos is None or last_pos is None:
        #no_frame_cont += 1
        velocity = -1
        #continue
    else:
        #calculate distances when numbers
        dist_x = current_pos[0] - last_pos[0]
        dist_y = current_pos[1] - last_pos[1]
        space = math.hypot(dist_x, dist_y)

        #print((str(current_pos[0]) + '-' + str(last_pos[0]) )+ ';' + str(space) )

        # TODO: total displacement divided number of frames where particle was not recognised
        #velocity = ((space/no_frame_cont) * 5.1 / 400) * fps
        velocity = ((space ) * 5.1 / 400) * fps
        #no_frame_cont = 1



        #print(velocity)

    # uncomment to save data
    string_to_write = str(current_pos) + ';' + str(velocity) + '\n'
    print(string_to_write)
    file.write(string_to_write)
    file.flush()

    #draw the trajectory use pts list to vanishing tail, use center to see all trajectories
    own_fns.drawTrajectory(center, im_with_keypoints)



    cv2.imshow('keypoints', im_with_keypoints)
    cv2.waitKey()



    #uncomment this to save images
    #cv2.imwrite(dir + 'saved-images/' + '{0:04}'.format(cont) + '.png', im_with_keypoints)
    #cv2.imshow("Frame", frame)


    cont += 1


    #frame1 = frame2

# close file and finish script
cap.release()
cv2.destroyAllWindows()
file.close()


