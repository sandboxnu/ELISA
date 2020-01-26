# import the necessary packages
import numpy as np
import argparse, cv2, imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

# load and resize the image
image = imutils.resize(cv2.imread(args["image"]), width=700)

# define the list of boundaries
boundaries = [
	([0, 0, 0], [82, 82, 82])
]

# loop over the boundaries
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    # if no black in image, then the mask will resemble the following array:
    # [[0 0 0 ... 0 0 0]
    #  [0 0 0 ... 0 0 0]
    #  ...
    #  [0 0 0 ... 0 0 0]]
    output = cv2.bitwise_and(image, image, mask = mask)

    # show the images
    cv2.imshow("images", np.hstack([image, output]))
    cv2.waitKey(0)
