import cv2
import numpy as np

class Blob:
    """
    Class blob to organize blobs and positions belonging to them\n
    region: positions conforming this blob\n
    centroid: mass centre from blob\n
    label: labelling number assigned to this blob\n
    blobChecked: boolean indicator if this blob was already revised by Blob.pointInBlob\n
    size: number of points conforming this blob\n
    lastTrackPos: stores the position of this blob on the last frame

    """
    def __init__(self, region, label):
        self.region = region
        #self.blobMap = blobMap
        #self.centroid = tuple(reversed(np.mean(self.region, axis=0)))
        self.centroid = tuple(np.mean(self.region, axis=0))
        self.label = label
        self.blobChecked = False
        self.size = len(self.region)
        self.lastTrackPos = None



    def pointInBlob(self, keypoint):
        """
        Determines if a given point belongs to this blob.\n
        :param keypoint: point to be analised \\
        :return: true or false if the analided keypoint belongs to the blob\n
        """
        kp = np.int_(np.array(keypoint))
        kp = tuple(kp)
        #check if exact point is inside the region
        if tuple(map(tuple, self.region)).count((kp)) > 0:
            self.blobChecked = True
            return True
        else:
            return False

    def showColouredBlob(self, drawing):
        """
        Draws the current blob\n
        :param drawing: image to draw the blob, normally an empty dark image with 3 channels\n
        :return: -1 -> It automatically shows the image on a new window
        """
        cv2.drawContours(drawing, [self.region], 0, (0,255,255), -1)
        cv2.imshow('drawn blob', drawing)
        cv2.waitKey()
        return -1

