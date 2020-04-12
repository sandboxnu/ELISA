# libraries
import cv2
import imutils
import numpy as np

# read in image
image = cv2.imread("../images/plate_black_1.jpeg")

# resized = imutils.resize(image, width=700)
# ratio = image.shape[0] / float(resized.shape[0])
#
# # convert the resized image to grayscale, blur it slightly,
# # and threshold it
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (11, 11), 0)
# thresh = cv2.threshold(blurred, 39, 255, cv2.THRESH_BINARY)[1]
# # for debugging:
# # cv2.imshow("thresh", thresh)
#
# # find contours in the thresholded image
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#     cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
#
# # initialize vars to track largest contour and largest contour area
# largestC = cnts[0]
# largestArea = 0
#
# # loop over the contours to find the one with the largest area
# for c in cnts:
#     # find area of contour
#     area = cv2.contourArea(c)
#     if area > largestArea :
#         largestArea = area
#         largestC = c
#
# # multiply the contour (x, y)-coordinates by the resize ratio,
# # then draw the contours on the image
# largestC = largestC.astype("float")
# largestC *= ratio
# largestC = largestC.astype("int")
# cv2.drawContours(image, [largestC], -1, (0, 255, 0), 2)
#
# # adapted from tutorial at https://hub.packtpub.com/opencv-detecting-edges-lines-shapes/
# # find minimum area
# rect = cv2.minAreaRect(largestC)
# # calculate coordinates of the minimum area rectangle
# box = cv2.boxPoints(rect)
#
# # rotate img
# angle = rect[2]
# rows, cols = image.shape[0], image.shape[1]
# M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
# img_rot = cv2.warpAffine(image, M, (cols, rows))
#
# # rotate bounding box
# rect0 = (rect[0], rect[1], 0.0)
# box = cv2.boxPoints(rect)
# pts = np.int0(cv2.transform(np.array([box]), M))[0]
# pts[pts < 0] = 0
#
# # crop
# img_crop = img_rot[pts[1][1]:pts[0][1],
#            pts[1][0]:pts[2][0]]
#
# cv2.imshow('Image', image)
# cv2.imshow('Final',img_crop)
# cv2.waitKey(0)

# convert to gray and find contours
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contours, h = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

final = np.zeros(image.shape, np.uint8)
mask = np.zeros(gray.shape,np.uint8)

# draw contours using mask, and show average color in contours
for i in range(0,len(contours)):
    mask[...]=0
    cv2.drawContours(mask,contours,i,255,-1)
    cv2.drawContours(final, contours, i, cv2.mean(image, mask), -1)

# resize image
scale_percent = 20 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image1 = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
final1 = cv2.resize(final, dim, interpolation = cv2.INTER_AREA)

# show images
cv2.imshow('Image', image1)
cv2.imshow('Final', final1)
cv2.waitKey(0)