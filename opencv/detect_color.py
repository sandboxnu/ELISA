# import the necessary packages
import numpy as np
import argparse, cv2, imutils
import sys

# adpated from tutorial at https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

# load and resize the image
image = imutils.resize(cv2.imread(args["image"]), width=700)

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
		sys.exit("Background must be black.")
	elif avg[1] > upperThresh[1]:
		sys.exit("Background must be black.")
	elif avg[2] > upperThresh[2]:
		sys.exit("Background must be black.")
	else:
		pass

# run avgColor on all four corners
topLeftAvg = avgColor(topRows, leftCols)
topRightAvg = avgColor(topRows, rightCols)
botLeftAvg = avgColor(botRows, leftCols)
botRightAvg = avgColor(botRows, rightCols)

# if code executes to this point, this means that the background of the image was within the defined boundaries
# for debugging:
print("Success! Image background is black.")
