import os
import cv2
import numpy as np
import math
#import matplotlib.pyplot as plt

#  use matplotlib for showing of images, otherwise error will be thrown since something is not properly installed/working
dir = os.getcwd()+'/examples/videos/'

#load video
cap = cv2.VideoCapture(dir + 'pong.avi')
#determine frames per second of loaded video
fps = cap.get(cv2.CAP_PROP_FPS)

#check if video is correctly loaded
if (cap.isOpened()== False):
    print("Error opening video file, check file location and name")

#to count frames
cont = 0

#uncomment if required to save data
file = open('datafile.txt', 'w')


while(cap.isOpened()):
    #pass current frame of video as RGB image to frame object
    ret, frame = cap.read()

    #to check if we are finished reading frames and break the loop
    if frame is None:
        print('All frames read')
        break

    #separate RGB matrix...remember opencv works in BGR!!!
    blue = frame[:, :, 0]
    green = frame[:, :, 1]
    red = frame[:, :, 2]

    #binary threshold for simple segmentation
    ret, ball1 = cv2.threshold(red, 20, 255, cv2.THRESH_BINARY)
    ret, ball2 = cv2.threshold(blue, 220, 255, cv2.THRESH_BINARY_INV)

    #binary operations to separate pixels of the balls from other elements in videos
    ball = cv2.bitwise_not(cv2.bitwise_or(ball1, ball2))

    #smoothing of circular shape
    ball = cv2.GaussianBlur(ball, (7, 7), 2, 2)

    #Hough circles function to detect ball in each frame
    circles = cv2.HoughCircles(ball, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 80, 20, 1, 300)


    if circles is not None:
        a, b, c = circles.shape
        for i in range(b):
            #draw a circle and its center on the original image
            cv2.circle(frame, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (circles[0][i][0], circles[0][i][1]), 1, (0, 0, 255), 1, cv2.LINE_AA)

            #current coordinates of the centroid
            x2 = circles[0][i][0]
            y2 = circles[0][i][1]

            #calculate distance moved (euclidean) with reference to previous frame
            if cont is not 0:
                dist_x = x2 - p_prev[0]
                dist_y = y2 - p_prev[1]
                dist = math.hypot(dist_x, dist_y)

            #show velocities corresponding to natural movements (ball not dissapearing)
            if cont is not 0 and dist_y != 0 and dist_x != 0 and dist < 500 :
                #uncomment to save data
                #string_to_write = str(dist_x*fps) + ';' + str(dist_y*fps) + ';' + str(dist) + '\n'
                #file.write(string_to_write)

                vel = dist*fps
                print('vel', '{0:.02f}'.format(vel), ' / vel x: ', dist_x*fps, ' / vel y: ', dist_y*fps)
                #uncommment to show  velocity on image
                cv2.putText(frame, "vel: " + str(int(vel)) + 'pix/sec', (circles[0][i][0], circles[0][i][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0, 255, 255), 2);
                cv2.putText(frame, "vel x: " + str(int(dist_x*fps)) + 'pix/sec', (circles[0][i][0], int(circles[0][i][1]+20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2);
                cv2.putText(frame, "vel y: " + str(int(dist_y*fps)) + 'pix/sec', (circles[0][i][0], int(circles[0][i][1]+40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2);



        #save centroid coordinate for next iteration computation
        p_prev = circles[0][i][0], circles[0][i][1]

    cont += 1

    #show image with corresponding frame
    cv2.putText(frame, 'frame ' + str(cont), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50),2);
    cv2.imshow('ball detected', frame)
    #plt.imshow('ball detected', frame) # does not show anything
    cv2.waitKey(0)

    #cv2.imwrite(dir + 'saved-images/' + '{0:04}'.format(cont) + '.jpg', frame)


#close file and finish script

cap.release()
#cv2.destroyAllWindows()
file.close()
