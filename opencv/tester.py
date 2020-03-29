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
#pi.check_background()

# test detect_bounds and draw_contours
#pi.draw_contours().show()
#cv2.waitKey(0)

# test normalize_shape
# pi.normalize_shape().show()
# cv2.waitKey(0)

# test get_vials using draw_vials
pi.normalize_shape().draw_vials()
cv2.waitKey(0)
