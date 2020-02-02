import argparse, cv2, imutils
from PlateImage import PlateImage

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load image and initialize plate image
image = cv2.imread(args["image"])
pi = PlateImage(image)

# test check_background
pi.check_background()

# test detect_bounds
cv2.imshow( "Detect Bounds Results", imutils.resize(pi.detect_bounds(), width=700) )
cv2.waitKey(0)
