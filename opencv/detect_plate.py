import argparse, cv2, imutils
import numpy as np

'''
NOTES:
- works with test images plate_black_1.jpeg -> plate_black_5.jpeg
- doesn't work if the background color is not black
'''

# adapted from tutorial at https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
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

# find minimum area
# adapted from tutorial at https://hub.packtpub.com/opencv-detecting-edges-lines-shapes/
rect = cv2.minAreaRect(largestC)
# calculate coordinates of the minimum area rectangle
box = cv2.boxPoints(rect)
# normalize coordinates to integers
box = np.int0(box)
# draw contours on the image
cv2.drawContours(image, [box], 0, (0,0, 255), 3)


# show the output image
cv2.imshow("Image", imutils.resize(image, width=700))
cv2.waitKey(0)
