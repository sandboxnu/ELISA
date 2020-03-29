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
        #self.image = image
        self.image = imutils.resize(image, width=700)

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

    # () -> ()
    # raises an error if the image background isn't black
    def check_background(self):
        # adpated from tutorial at https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

        # copy the image and resize the copy
        image = self.image.copy()

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

    # () -> Array of coordinates for rectangle corners
    # determines the smallest rectangle from the detected edges of the plate
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

        # adapted from tutorial at https://hub.packtpub.com/opencv-detecting-edges-lines-shapes/
        # find minimum area
        rect = cv2.minAreaRect(largestC)

        return rect

    # () -> New image with drawn contours
    # draws the detected boundaries onto the image
    def draw_contours(self):

        rect = self.detect_bounds()
        image_with_contours = self.image.copy()

        # calculate coordinates of the minimum area rectangle
        box = cv2.boxPoints(rect)
        # normalize coordinates to integers
        box = np.int0(box)
        # draw contours on the image
        cv2.drawContours(image_with_contours, [box], 0, (0, 0, 255), 3)

        # return the copied image with the contours drawn onto it
        return PlateImage(image_with_contours)

    # () -> [[(row, col, radius)]]
    # generates a list of the vial locations
    # returns the center positions and radii (half side length of square) of the areas from which we will read the colors
    def get_vials(self):
        dims = self.image.shape
        center = ((dims[0]) // 2, (dims[1]) // 2)
        return [ # Row 7
                (center[0] + 28, center[1] - 35, 10), #E7
                (center[0] + 28, center[1] + 35, 10), #D7
                (center[0] + 28, center[1] - 110, 10), #F7
                (center[0] + 28, center[1] + 110, 10), #C7
                (center[0] + 28, center[1] - 185, 10), #G7
                (center[0] + 28, center[1] + 185, 10), #B7
                (center[0] + 28, center[1] - 260, 10), #H7
                (center[0] + 28, center[1] + 260, 10), #A7
                # Row 6
                (center[0] - 50, center[1] - 35, 10), #E6
                (center[0] - 50, center[1] + 35, 10), #D6
                (center[0] - 50, center[1] - 110, 10), #F6
                (center[0] - 50, center[1] + 110, 10), #C6
                (center[0] - 50, center[1] - 185, 10), #G6
                (center[0] - 50, center[1] + 185, 10), #B6
                (center[0] - 50, center[1] - 260, 10), #H6
                (center[0] - 50, center[1] + 260, 10), #A6
                # Row 5
                (center[0] - 120, center[1] - 35, 10), #E5
                (center[0] - 120, center[1] + 35, 10), #D5
                (center[0] - 120, center[1] - 110, 10), #F5
                (center[0] - 120, center[1] + 110, 10), #C5
                (center[0] - 120, center[1] - 185, 8), #G5
                (center[0] - 120, center[1] + 185, 8), #B5
                (center[0] - 120, center[1] - 260, 8), #H5
                (center[0] - 120, center[1] + 260, 8), #A5
                # Row 4
                (center[0] - 200, center[1] - 35, 10), #E4
                (center[0] - 200, center[1] + 35, 10), #D4
                (center[0] - 200, center[1] - 110, 10), #F4
                (center[0] - 200, center[1] + 110, 10), #C4
                (center[0] - 200, center[1] - 185, 8), #G4
                (center[0] - 200, center[1] + 185, 8), #B4
                (center[0] - 200, center[1] - 260, 8), #H4
                (center[0] - 200, center[1] + 260, 8), #A4
                # Row 3
                (center[0] - 280, center[1] - 35, 10), #E3
                (center[0] - 280, center[1] + 35, 10), #D3
                (center[0] - 280, center[1] - 110, 10), #F3
                (center[0] - 280, center[1] + 110, 10), #C3
                (center[0] - 280, center[1] - 185, 8), #G3
                (center[0] - 280, center[1] + 185, 8), #B3
                (center[0] - 280, center[1] - 260, 8), #H3
                (center[0] - 280, center[1] + 260, 8), #A3
                # Row 2
                (center[0] - 350, center[1] - 35, 9), #E2
                (center[0] - 350, center[1] + 35, 9), #D2
                (center[0] - 350, center[1] - 110, 9), #F2
                (center[0] - 350, center[1] + 110, 9), #C2
                (center[0] - 350, center[1] - 185, 8), #G2
                (center[0] - 350, center[1] + 185, 8), #B2
                (center[0] - 350, center[1] - 260, 8), #H2
                (center[0] - 350, center[1] + 260, 8), #A2
                # Row 1
                (center[0] - 415, center[1] - 35, 9), #E1
                (center[0] - 415, center[1] + 35, 9), #D1
                (center[0] - 415, center[1] - 110, 9), #F1
                (center[0] - 415, center[1] + 110, 9), #C1
                (center[0] - 415, center[1] - 185, 8), #G1
                (center[0] - 415, center[1] + 185, 8), #B1
                (center[0] - 415, center[1] - 260, 6), #H1
                (center[0] - 415, center[1] + 250, 6), #A1
                # Row 8
                (center[0] + 100, center[1] - 35, 10), #E8
                (center[0] + 100, center[1] + 35, 10), #D8
                (center[0] + 100, center[1] - 110, 10), #F8
                (center[0] + 100, center[1] + 110, 10), #C8
                (center[0] + 100, center[1] - 185, 8), #G8
                (center[0] + 100, center[1] + 185, 8), #B8
                (center[0] + 100, center[1] - 260, 8), #H8
                (center[0] + 100, center[1] + 260, 8), #A8
                # Row 9
                (center[0] + 175, center[1] - 35, 10), #E9
                (center[0] + 175, center[1] + 35, 10), #D9
                (center[0] + 175, center[1] - 110, 10), #F9
                (center[0] + 175, center[1] + 110, 10), #C9
                (center[0] + 175, center[1] - 185, 8), #G9
                (center[0] + 175, center[1] + 185, 8), #B9
                (center[0] + 175, center[1] - 260, 8), #H9
                (center[0] + 175, center[1] + 260, 8), #A9
                # Row 10
                (center[0] + 250, center[1] - 35, 10), #E10
                (center[0] + 250, center[1] + 35, 10), #D10
                (center[0] + 250, center[1] - 110, 10), #F10
                (center[0] + 250, center[1] + 110, 10), #C10
                (center[0] + 250, center[1] - 185, 8), #G10
                (center[0] + 250, center[1] + 190, 8), #B10
                (center[0] + 250, center[1] - 265, 8), #H10
                (center[0] + 250, center[1] + 265, 8), #A10
                # Row 11
                (center[0] + 330, center[1] - 35, 9), #E11
                (center[0] + 330, center[1] + 35, 9), #D11
                (center[0] + 330, center[1] - 110, 9), #F11
                (center[0] + 330, center[1] + 110, 9), #C11
                (center[0] + 330, center[1] - 185, 8), #G11
                (center[0] + 330, center[1] + 190, 8), #B11
                (center[0] + 330, center[1] - 265, 8), #H11
                (center[0] + 330, center[1] + 265, 8), #A11
                # Row 12
                (center[0] + 410, center[1] - 35, 9), #E12
                (center[0] + 410, center[1] + 35, 9), #D12
                (center[0] + 410, center[1] - 110, 9), #F12
                (center[0] + 410, center[1] + 110, 9), #C12
                (center[0] + 410, center[1] - 185, 8), #G12
                (center[0] + 410, center[1] + 190, 8), #B12
                (center[0] + 410, center[1] - 265, 6), #H12
                (center[0] + 410, center[1] + 265, 6), #A12
                ]

    # () -> New image with important pixels highlighted
    # changes the pixel color of centers of the optimal parts of the vials and the color of four pixels around them based on the radius for testing the accuracy of get_vials
    def draw_vials(self):
        vials = self.get_vials()
        image = self.image.copy()
        for vial in vials:
            # highlight pixel at center
            image[vial[0]][vial[1]] = [0, 0, 255]
            # also turn its neighboring pixels red for better visualization
            image[vial[0]+1][vial[1]+1] = [0, 0, 255]
            image[vial[0]+1][vial[1]] = [0, 0, 255]
            image[vial[0]+1][vial[1]-1] = [0, 0, 255]
            image[vial[0]][vial[1]+1] = [0, 0, 255]
            image[vial[0]][vial[1]-1] = [0, 0, 255]
            image[vial[0]-1][vial[1]+1] = [0, 0, 255]
            image[vial[0]-1][vial[1]] = [0, 0, 255]
            image[vial[0]-1][vial[1]-1] = [0, 0, 255]
            # draw the corners of the square based on the radius in blue
            # highligh their neighbors are well for better visualization
            top = vial[0]-vial[2]
            bottom = vial[0]+vial[2]
            left = vial[1]-vial[2]
            right = vial[1]+vial[2]
            # top left
            image[top][left] = [255, 0, 0]
            image[top+1][left+1] = [255, 0, 0]
            image[top+1][left] = [255, 0, 0]
            image[top+1][left-1] = [255, 0, 0]
            image[top][left] = [255, 0, 0]
            image[top][left] = [255, 0, 0]
            image[top-1][left+1] = [255, 0, 0]
            image[top-1][left] = [255, 0, 0]
            image[top-1][left-1] = [255, 0, 0]

            # top right
            image[top][right] = [255, 0, 0]
            image[top+1][right+1] = [255, 0, 0]
            image[top+1][right] = [255, 0, 0]
            image[top+1][right-1] = [255, 0, 0]
            image[top][right] = [255, 0, 0]
            image[top][right] = [255, 0, 0]
            image[top-1][right+1] = [255, 0, 0]
            image[top-1][right] = [255, 0, 0]
            image[top-1][right-1] = [255, 0, 0]

            # bottom left
            image[bottom][left] = [255, 0, 0]
            image[bottom+1][left+1] = [255, 0, 0]
            image[bottom+1][left] = [255, 0, 0]
            image[bottom+1][left-1] = [255, 0, 0]
            image[bottom][left] = [255, 0, 0]
            image[bottom][left] = [255, 0, 0]
            image[bottom-1][left+1] = [255, 0, 0]
            image[bottom-1][left] = [255, 0, 0]
            image[bottom-1][left-1] = [255, 0, 0]

            # bottom right
            image[bottom][right] = [255, 0, 0]
            image[bottom+1][right+1] = [255, 0, 0]
            image[bottom+1][right] = [255, 0, 0]
            image[bottom+1][right-1] = [255, 0, 0]
            image[bottom][right] = [255, 0, 0]
            image[bottom][right] = [255, 0, 0]
            image[bottom-1][right+1] = [255, 0, 0]
            image[bottom-1][right] = [255, 0, 0]
            image[bottom-1][right-1] = [255, 0, 0]

        cv2.imshow("Vial locations", imutils.resize(image, width=500))

    # () -> [[color]]
    # reads the colors of each vial on the plate
    # returns some data structure to demonstrate the array
    def get_colors(self):
        raise "not yet implemented"

    # (maybe path) -> () (saves image)
    # read the colors of the image and export image to paht
    def export_colors(self, path = "."):
        raise "not yet implemented"

    # () -> New cropped image
    # normalize the image to have a rectangular shape
    def normalize_shape(self):

        rect = self.detect_bounds()
        image = self.image.copy()
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

        return PlateImage(img_crop)

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
