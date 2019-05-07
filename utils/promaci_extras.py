import cv2
import numpy as np
import os




def CropImage(im, bbox):
    """
    Crop an image based on a box given by the selectROI function\n

    im: image\n
    bbox: bounding box selected previously by cv2.selectROI\n
    """
    im = im[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
    return im


def drawTrajectory(pts, im):
    """
    Draw the points based on a list of (x,y) coordinates to in image, cv2.polylines can also be used.\n

    pts: (x,y) list from points to draw\n
    im: image where the trajectory will be coloured \n
    Change the BGR(255, 255, 255) parameter to draw on a different colour\n
    """
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        cv2.line(im, pts[i - 1], pts[i], (255, 255, 255), 1)



def draw_str(dst, target, s):
    """
    Draw a string on an image\n

    dst: image to write string\n
    target: position where string will be drawn, remember that origin is on the upper left part of the image\n
    s: string to be drawn\n
    """
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)



def videoToImgs(dir, video_name, video_extension, im_extension):
    """
    Convert all frames of a video into images and save them on a given folder, it automatically creates the new folder\n

    dir: directory where images will be saved\n
    video_name: name of video without extension\n
    video_extension: string with the extension of the video, without the point (.), e.g. "avi"\n
    im_extension: string with the desired extensions for the imates, without the point (.), e.g. "png"\n
    """
    new_folder = os.path.join(dir, video_name)

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    cap = cv2.VideoCapture(os.path.join(new_folder + '.' + video_extension))

    print ('Reading and saving frames')

    n = 1
    n_empty = 0

    while (cap.isOpened()):

        ret, frame = cap.read()
        cv2.imwrite(os.path.join(new_folder, '{0:04}'.format(n) + '.' + im_extension), frame)

        if frame is None:
            print ('Frame ' + str(n) + ' was empty')
            n_empty +=1
        if n == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            print ('Printed frames: ' + str(n-n_empty) + ' out of ' + str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            break
        n += 1

    cap.release()
    print('Conversion finished')

def blobFeatures(bin):


    # Set up the SimpleBlobdetector with default parameters.-------------------
    """
    SimpleBlobDetedtor works by doing different thresholds to the iamge (let's say every 10 levels of intensity) between
    a min and a max thresh value, with this it is determined which connected regions are present in most of the thresholds done
    and therefore are more likely to be our actual particles, They can be then filtered by size or shape with further parameters\n

    In our case, the simple blob detector gets a binary image, so it is mostly used to filter blobs by the different parameters.\n

    It returns the kepoints of the DETECTED blobs. So we use it as detector.

    """

    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255


    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.5

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.2


    # Filter by Area.
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 5000


    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.1

    detector = cv2.SimpleBlobDetector_create(params)
    # -----------------------------------------------------------------------

    keypoints = detector.detect(255 - bin)
    return keypoints



def imageBinarization(img):
    """
    Produces the binary image by applying gaussian blurring, threshold, erosion and opening\n

    img: initial image from video\n
    bin: resulting binary image\n
    blur_img: grayscale image with median blurring\n
    The morphological operations can be modified for improved results, the threshold is done via Otsu, i.e it searchs the best threshold value automatically.

    """

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.medianBlur(img_gray, 5)

    retval2, bin = cv2.threshold(blur_img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    bin = cv2.erode(bin, kernel)
    bin = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)

    return bin, blur_img


