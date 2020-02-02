#!/usr/bin/env python3
import imutils
import cv2
import numpy as np

# Represents an image of an ELISA plate
class PlateImage:

    # default constructor
    # creates image from provided image path
    # when adding parameters (such as thresholds),
    # these should be optional parameters for methods
    def __init__(self, image):
        # TODO: determine what standard size to reduce the stored image to
        self.image = image

    def from_path(self, imagePath):
        return self.__init__(cv2.imread(imagePath))

    # () -> Boolean
    # determines whether the image is blurry
    def is_blurry(self):
        raise "not yet implemented"

    # () -> ()
    # raises an error if the image background isn't black
    def check_background(self):
        # adpated from tutorial at https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

        # copy the image and resize the copy
        image = self.image.copy()
        image = imutils.resize(image, width=700)

        # define the coordinates of the corners of the image
        num_rows, num_cols, num_dims = image.shape
        topRows = range(25)
        botRows = range( (num_rows-25), num_rows )
        leftCols = range(25)
        rightCols = range( (num_cols-25), num_cols )

        # define color boundaries - this is the range of our target background colors
        lowerThresh = [0, 0, 0]
        upperThresh = [51, 51, 51]

        # find the average color value of the pixels in a given corner of the image and throw an error if they are lighter than the upperTresh
        def avgColor(rows, cols):
        	totalRed = 0
        	totalGreen = 0
        	totalBlue = 0
        	for r in rows:
        		for c in rows:
        			totalRed += image[r,c][0]
        			totalGreen += image[r,c][1]
        			totalBlue += image[r,c][2]
        	avg = np.array([(totalRed/625), (totalGreen/625), (totalBlue/625)])
        	# check to see if each component of the rgb value is darker than the upper boundary, otherwise throw an error
        	if avg[0] > upperThresh[0]:
        		raise ValueError("Background must be black.")
        	elif avg[1] > upperThresh[1]:
        		raise ValueError("Background must be black.")
        	elif avg[2] > upperThresh[2]:
        		raise ValueError("Background must be black.")
        	else:
        		pass

        # run avgColor on all four corners
        topLeftAvg = avgColor(topRows, leftCols)
        topRightAvg = avgColor(topRows, rightCols)
        botLeftAvg = avgColor(botRows, leftCols)
        botRightAvg = avgColor(botRows, rightCols)

        # if code executes to this point, this means that the background of the image was within the defined boundaries
        # for debugging:
        # print("Success! Image background is black.")

    # () -> () (mutates object)
    # generating a new image from this one
    def detect_bounds(self):
        # adapted from tutorial at https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/

        # copy the image and resize the copy to a smaller factor so that
        # the shapes can be approximated better
        image = self.image.copy()
        resized = imutils.resize(image, width=700)
        ratio = image.shape[0] / float(resized.shape[0])

        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        thresh = cv2.threshold(blurred, 39, 255, cv2.THRESH_BINARY)[1]
        # for debugging:
        # cv2.imshow("thresh", thresh)

        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # initialize vars to track largest contour and largest contour area
        largestC = cnts[0]
        largestArea = 0

        # loop over the contours to find the one with the largest area
        for c in cnts:
        	# find area of contour
        	area = cv2.contourArea(c)
        	if area > largestArea :
        		largestArea = area
        		largestC = c

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours on the image
        largestC = largestC.astype("float")
        largestC *= ratio
        largestC = largestC.astype("int")
        cv2.drawContours(image, [largestC], -1, (0, 255, 0), 2)

        # adapted from tutorial at https://hub.packtpub.com/opencv-detecting-edges-lines-shapes/
        # find minimum area
        rect = cv2.minAreaRect(largestC)
        # calculate coordinates of the minimum area rectangle
        box = cv2.boxPoints(rect)
        # normalize coordinates to integers
        box = np.int0(box)
        # draw contours on the image
        cv2.drawContours(image, [box], 0, (0,0, 255), 3)

        # return the copied image with the contours drawn onto it
        return image

    # () -> [[color]]
    # reads the colors of each vial on the plate
    # returns some data structure to demonstrate the array
    def get_colors(self):
        raise "not yet implemented"

    # (maybe path) -> () (saves image)
    # read the colors of the image and export image to paht
    def export_colors(self, path = "."):
        raise "not yet implemented"

    # () -> () (mutates object)
    # normalize the image to have a rectangular shape
    def normalize_shape(self):
        raise "not yet implemented"

    # () -> () (mutates object)
    # normalize the colors of the image to improve color accuracy
    def normalize_color(self):
        raise "not yet implemented"

    # (maybe path) -> ()
    # writes the image to disk
    def save(self, path = "."):
        cv2.imwrite(path, self.image)

    # displays image through opencv
    # () -> ()
    def show(self):
        cv2.imshow("ELISA Plate", self.image)
