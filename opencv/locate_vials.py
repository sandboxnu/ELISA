import argparse, cv2, imutils
import numpy as np

'''
THINGS TO TEST:
- what is the difference between the different kinds of thresholding?
- adjusting adaptiveThreshold
- hough circles

THINGS ALREADY TRIED THAT WERE UNREALIABLE
- Using cv2.threshold with a preset threshold
    This may work well for some images, but there isn't any one optimal threshold value that produces feasable results for all images
- Using cv2.threshold with the THRESH_OTSU flag, which will use Otsu's binarization algorithm to find the optimal threshold for the image
    The thresholds used are always too high
- Inverting the image colors
    Isn't helpful
- Blurring, thresholding, and then graying the image instead of graying, blurring and then thresholding
    findContours picks up no contours on the resulting image
- Blurring, thresholding, graying, and then thresholding again
    Thresholding the second time doesn't work, no matter how low we set the threshold value
- Using adaptiveThreshold instead of threshold
    Doesn't make a difference
- Using canny edge detection to see if it makes finding contours more accurate
    Gives us more contours to work with, but overall much more inaccurate

ISSUES
- HoughCircles are the most "accurate" (even then, they're not very accurate), but return different results when the image is cropped
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

'''
# use Canny edge detection to find the edges of the image
edges = cv2.Canny(resized, 100, 200)
cv2.imshow("edges", edges)

# imagem = cv2.bitwise_not(resized)
# cv2.imshow("inverted", imagem)
'''

# convert the resized image to grayscale, blur it slightly, and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
gray_thresh = cv2.threshold(gray_blur, 121, 255, cv2.THRESH_BINARY)[1]
#gray_threshed = #cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY,11,2)
# otso, gray_threshed = cv2.threshold(gray_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# print(otso)
'''
# blur the resized image, blur it, threshold it, and then convert the threshold to grayscale
blurred = cv2.GaussianBlur(resized, (15, 15), 0)
thresh = cv2.threshold(blurred, 61, 255, cv2.THRESH_BINARY)[1]
thresh_gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
rethresh = cv2.threshold(thresh_gray, 1, 255, cv2.THRESH_BINARY)[1]
'''
# for debugging:
cv2.imshow("thresh gray", gray_thresh)
# cv2.imshow("thresh color", thresh)
# cv2.imshow("thresh gray", thresh_gray)
# cv2.imshow("rethresh", rethresh)


# find contours in the thresholded image
cnts = cv2.findContours(gray_thresh.copy(), cv2.RETR_LIST,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# initialize vars to track vial contours
vials = []

# loop over the contours to find the ones with an appropriately size area, and put them into a list
for c in cnts:
	# find area of contour
	area = cv2.contourArea(c)
	if area > 0 :
		vials.append(c)

# multiply the contours (x, y)-coordinates by the resize ratio,
# then draw the contours on the image
for vial in vials:
    vial = vial.astype("float")
    vial *= ratio
    vial = vial.astype("int")
    cv2.drawContours(image, [vial], -1, (0, 255, 0), 2)

# show the output image
cv2.imshow("Image", imutils.resize(image, width=700))

'''
# find minimum area
# adapted from tutorial at https://hub.packtpub.com/opencv-detecting-edges-lines-shapes/
rect = cv2.minAreaRect(largestC)
# calculate coordinates of the minimum area rectangle
box = cv2.boxPoints(rect)
# normalize coordinates to integers
box = np.int0(box)
# draw contours on the image
cv2.drawContours(image, [box], 0, (0,0, 255), 3)
'''


circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=30)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(gray,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(gray,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
