
import cv2
import imutils
import numpy as np
import math
from glob import glob
from utils import promaci_extras
from utils import classBlob as Blob
import os
from imutils import contours

import random


#name of main directory, video to use and set frames per second
dir = '/home/jmateo/Documents/optrack'
video_name = "50 Hz Belichtungszeit 5000-1"

cap = cv2.VideoCapture(os.path.join(dir,video_name + '.avi'))

if (cap.isOpened() == False):
    print("Error opening video file, check file location and name")


#fps = 30
videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print('video length: ', str(videoLength))
print('fps: ', str(fps))

#uncomment this lines when reading images from a folder
#frames_list = sorted(glob(os.path.join(dir, video_name, '*.png')))
#frame0= cv2.imread(frames_list[0])

ret, frame0 = cap.read()

#select ROI and save it to crop every image equally

bbox = cv2.selectROI(frame0, False)
cv2.destroyWindow('ROI selector')
frame0 = promaci_extras.CropImage(frame0, bbox)

#image binarization and blurring with own algorithm
bin0, blur_img0 = promaci_extras.imageBinarization(frame0)
h,w = bin0.shape


initial_frame = 1

#parameters for Lukas-Kanade optical flow tracking, see opencv documentation for further explanation.
lk_params = dict(winSize=(15, 15),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.0001))

#color = np.random.randint(0,255,(100,3))
#mask = np.zeros_like(background)

old = blur_img0
# old = mask
tracks = []
centroids = []

track_len = 50
frame_idx = 0
detect_interval = 3

#vector to output the velocities on a file, normally its number of rows = particles as there are
# add a few more to avoid dimension problems when having noise in image , reflexions etc.

arrayVel = np.zeros([videoLength, 300])
arrY = np.zeros([videoLength, 300])
arrX = np.zeros([videoLength, 300])

#Use the for and the following command, instead of the while loop and the cap.read, when working with images from folder.
#for n in frames_list[initial_frame:]:
    #frame = cv2.imread(n)
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        print ('All frames read')
        break

    frame = promaci_extras.CropImage(frame, bbox)

    bin, blur_img = promaci_extras.imageBinarization(frame)

    vis = frame.copy()
    blobMap = np.zeros_like(bin)
    #get an image with all blobs eunmerated
    cnts = cv2.findContours(bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = contours.sort_contours(cnts)[0]

    #for each contour:
    #create an instance and add it to the list of blobs
    #create label its whole region on an image (0 to len(contour))
    blobList = []
    label = 200
    label2 = 200
    drawing = np.zeros_like(bin)
    for i in cnts:
        #cnt = cnts[n]
        blobMap = np.zeros_like(bin)
        cv2.drawContours(blobMap, [i], 0, label, -1)
        #cv2.imshow('drawn contour', blobMap)

        positions = np.argwhere(cv2.transpose(blobMap) == label)
        if len(positions) > 5 and len(positions) < 1500:
            #create respective blob with the positions and label assigned
            b = Blob(positions, label)
            cv2.drawContours(drawing, [i], 0, 1, -1)
            #print(b.size)
            #append the created blob to the list of blobs of this frame
            blobList.append(b)
            label += 1
            # b.showColouredBlob(frame.copy())
        label2 += 1

    # perform point-wise multiplication to mask the resulting blobs
    mask = np.multiply(blur_img, drawing)


    if len(tracks) > 0:
        # optical flow Lukas-Kanade with back-tracking for match verification between frames

        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(old, mask, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(mask, old, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        #This distance can be changed to allow more possible matches.
        good = d < 20

        new_tracks = []

        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            # if len(tr) > track_len:
            #     del tr[0]
            new_tracks.append(tr)
            cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
        tracks = new_tracks

        #colour and print all the results on the images
        cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (255, 0, 0))
        #use random colour for each detected trajectory
        colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.polylines(frame0, [np.int32(tr) for tr in tracks], False, colour)
        promaci_extras.draw_str(vis, (20, h - 20), 'Frame %d' % frame_idx)
        promaci_extras.draw_str(vis, (20, h-40), 'Blobs tracked: %d' % len(tracks))

        #calculate velocity of particles
        for tr in range(len(tracks)):
            #get the current and last tracked points
            currentVel = tracks[tr][-1]
            lastVel = tracks[tr][-2]
            #separate the x and y positions
            dist_x = currentVel[0] - lastVel[0]
            dist_y = currentVel[1] - lastVel[1]
            # calculate distances with obtained positions
            #euclidean distance calculation
            space = math.hypot(dist_x, dist_y)
            #conversion to mm/s
            velocity = (space * 5.1 / 400)*fps
            #filter when velocity is very low, it may not be a particle, so assume it's noise and not take it into account
            if velocity > 0.1:
                arrayVel[frame_idx-1][tr] = velocity
                arrX[frame_idx-1][tr] = currentVel[0]
                arrY[frame_idx-1][tr] = currentVel[1]


    #blob detection every "detect_interval" value
    if frame_idx % detect_interval == 0:
        #obtain the keypoints from blob detector
        kp = promaci_extras.blobFeatures(bin)
        #print('keypoints detected:' + str(len(kp)))

        #if keypoints were not empty
        if kp is not None:

            #if tracks is not on the first iteration
            if len(tracks) > 0:
                #unpack last points from tracks in last_tracks
                last_tracks = []
                n = 0
                for l in range(len(tracks)):
                    last_tracks.append(tracks[l][-1])

                # map last track members to blobs
                for bl in blobList:
                    for l in range(len(last_tracks)):
                        if bl.pointInBlob(last_tracks[l]) == True:
                            bl.lastTrackPos = l
                            break

                #now check for each kp
                for n in range(len(kp)):
                    for bl in blobList:
                        kpAdded = False
                        #is kp on this blob and is this blob and old one (because no lastTrackPos is None so no mapping to tracks)
                        if bl.pointInBlob(kp[n].pt) == True and bl.lastTrackPos is not None:
                            #if so then make kp the new last center for this blob (center update)
                            tracks[bl.lastTrackPos][-1] = kp[n].pt
                            kpAdded = True
                            # go to next kp
                            break
                    #otherwise kp is from a new blob so add as new list at the end of tracks because kp was never added
                    if kpAdded == False:
                        tracks.append([kp[n].pt])

            else:
                #when tracks is an empty list add all keypoints
                for n in range(len(kp)):
                    tracks.append([kp[n].pt])

    #print('track # ' + str(len(tracks)))


    frame_idx += 1
    # old = blur_img
    old = mask


    # # show result, uncomment for observing the other images
    #cv2.imshow('blur_img', blur_img)
    cv2.imshow('lk_track', vis)
    # cv2.imshow('otsu + morpho', bin)
    # cv2.imshow('mask', mask)
    cv2.waitKey()

    #uncomment when needed to save each frame for posterior video creation
    #cv2.imwrite(dir + '/processed/' + '{0:04}'.format(cont) + 'png', vis)


    # cv2.imwrite(dir + '/processed/' + 'sample image_blur' +'.png', blur_img)
    # cv2.imwrite(dir + '/processed/' + 'sample image_bin' +'.png', bin)
    # cv2.imwrite(dir + '/processed/' + 'sample image' +'.png', frame0)

# Uncomment to save the positions and velocities measured
# np.savetxt(dir + '/processed/test.out', arrayVel, delimiter=';')
# np.savetxt(dir + '/processed/testX.out', arrX, delimiter=';')
# np.savetxt(dir + '/processed/testY.out', arrY, delimiter=';')


print('finished :)')
#uncomment to save the image with all the trajectories drawn
#cv2.imwrite(dir + '/'+video_name+' lines.png', frame0)

cv2.imshow('lk_track', frame0)
cv2.waitKey()


