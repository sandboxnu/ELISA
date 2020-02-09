import cv2
import imutils
import numpy as np

# https://www.quora.com/How-can-I-detect-an-object-from-static-image-and-crop-it-from-the-image-using-openCV
# https://stackoverflow.com/questions/28759253/how-to-crop-the-internal-area-of-a-contour

# read in image
image = cv2.imread("../images/plate_black_1.jpeg")

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
# box = cv2.boxPoints(rect)

# rotate img
angle = rect[2]
rows, cols = image.shape[0], image.shape[1]
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
img_rot = cv2.warpAffine(image, M, (cols, rows))

# rotate bounding box
rect0 = (rect[0], rect[1], 0.0)
box = cv2.boxPoints(rect)
pts = np.int0(cv2.transform(np.array([box]), M))[0]
pts[pts < 0] = 0

# crop
img_crop = img_rot[pts[1][1]:pts[0][1],
           pts[1][0]:pts[2][0]]

cv2.imshow('Image', image)
cv2.imshow('Final',img_crop)
cv2.imwrite('ROI.png',img_crop)
cv2.waitKey(0)


# image = cv2.imread('C:/Users/arkin/OneDrive/Arkin/Sandbox/Projects/ELISA/images/plate_black_1.jpeg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
#
# # Find contour and sort by contour area
# cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#
# # Find bounding box and extract ROI
# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
#     ROI = image[y:y+h, x:x+w]
#     break
#
# cv2.imshow('ROI',ROI)
# cv2.imwrite('ROI.png',ROI)
# cv2.waitKey()