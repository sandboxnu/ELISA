#!/usr/bin/env python3
import imutils
import cv2

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
    def is_blurry(self, threshold = 200):
        def variance_of_laplacian(grayImage):
            # compute the Laplacian of the image and then return the focus
            # measure, which is simply the variance of the Laplacian
            return cv2.Laplacian(grayImage, cv2.CV_64F).var()

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)

        return fm < threshold

    # () -> () (mutates object)
    # generating a new image from this one
    def detect_bounds(self):
        raise "not yet implemented"

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
