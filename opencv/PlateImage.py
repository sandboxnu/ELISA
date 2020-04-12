#!/usr/bin/env python3
import imutils
import cv2
import numpy as np
import scipy.cluster
from colorthief import ColorThief

# Represents an image of an ELISA plate
class PlateImage:

    # default constructor
    # creates image from provided image path
    # when adding parameters (such as thresholds),
    # these should be optional parameters for methods
    # ASSUME image is always scaled to have width of 700
    # TODO: 500 or 700??
    def __init__(self, image):
        self.image = imutils.resize(image, width=700)

    # creates a PlateImage given a path
    def from_path(self, image_path):
        return self.__init__(cv2.imread(image_path))

    # () -> Boolean
    # determines whether the image is blurry
    def is_blurry(self, threshold=200):

        def variance_of_laplacian(gray_image):
            # compute the Laplacian of the image and then return the focus
            # measure, which is simply the variance of the Laplacian
            return cv2.Laplacian(gray_image, cv2.CV_64F).var()

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)

        return fm < threshold

    # () -> ()
    # raises an error if the image background isn't black
    # upper_thresh, lower_thresh: range of target background colors
    def check_background(
            self, lower_thresh=[0, 0, 0], upper_thresh=[51, 51, 51]):
        # adapted from tutorial at
        # https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

        # copy the image and resize the copy
        image = self.image.copy()

        # define the coordinates of the corners of the image
        num_rows, num_cols, num_dims = image.shape
        top_rows = range(25)
        bot_rows = range((num_rows-25), num_rows)
        left_cols = range(25)
        right_cols = range((num_cols-25), num_cols)

        # find the average color value of the pixels in a given corner of image
        # throw an error if they are lighter than the upperTresh
        def avg_color(rows, cols):
            total_red = 0
            total_green = 0
            total_blue = 0
            for r in rows:
                for c in cols:
                    total_red   += image[r, c][0]
                    total_green += image[r, c][1]
                    total_blue  += image[r, c][2]
            avg = np.array([(total_red/625),
                            (total_green/625),
                            (total_blue/625)]) # TODO magic numbers

            # check to see if each component of the rgb value is darker than
            # upper boundary, otherwise throw an error
            for avg_color, thresh in avg, upper_thresh:
                if avg_color > thresh:
                    raise ValueError("Background must be black.")

        # run avg_color on all four corners
        avg_color(top_rows, left_cols)
        avg_color(top_rows, right_cols)
        avg_color(bot_rows, left_cols)
        avg_color(bot_rows, right_cols)

        # if code executes to this point, this means that the background
        # of the image was within the defined boundaries

       
    # () -> Array of coordinates for rectangle corners
    # determines the smallest rectangle from the detected edges of the plate
    def detect_bounds(self):
        # adapted from tutorial at
        # https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/

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
        largest_c = cnts[0]
        largest_area = 0

        # loop over the contours to find the one with the largest area
        for c in cnts:
            # find area of contour
            area = cv2.contourArea(c)
            if area > largest_area:
                largest_area = area
                largest_c = c

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours on the image
        largest_c = largest_c.astype("float")
        largest_c *= ratio
        largest_c = largest_c.astype("int")

        # adapted from tutorial at
        # https://hub.packtpub.com/opencv-detecting-edges-lines-shapes/
        # find minimum area
        rect = cv2.minAreaRect(largest_c)

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
    # returns the center positions and radii (half side length of square)
    # of the areas from which we will read the colors
    def get_vials(self):
        dims = self.image.shape
        center = ((dims[0]) // 2, (dims[1]) // 2)

        return [  # Row 7
                (center[1] - 35, center[0] + 28, 10),   # E7
                (center[1] + 35, center[0] + 28, 10),   # D7
                (center[1] - 110, center[0] + 28, 10),  # F7
                (center[1] + 110, center[0] + 28, 10),  # C7
                (center[1] - 185, center[0] + 28, 10),  # G7
                (center[1] + 185, center[0] + 28, 10),  # B7
                (center[1] - 260, center[0] + 28, 10),  # H7
                (center[1] + 260, center[0] + 28, 10),  # A7
                # Ro w  6
                (center[1] - 35, center[0] - 50, 10),   # E6
                (center[1] + 35, center[0] - 50, 10),   # D6
                (center[1] - 110, center[0] - 50, 10),  # F6
                (center[1] + 110, center[0] - 50, 10),  # C6
                (center[1] - 185, center[0] - 50, 10),  # G6
                (center[1] + 185, center[0] - 50, 10),  # B6
                (center[1] - 260, center[0] - 50, 10),  # H6
                (center[1] + 260, center[0] - 50, 10),  # A6
                # Row 5
                (center[1] - 35, center[0] - 120, 10),  # E5
                (center[1] + 35, center[0] - 120, 10),  # D5
                (center[1] - 110, center[0] - 120, 10), # F5
                (center[1] + 110, center[0] - 120, 10), # C5
                (center[1] - 185, center[0] - 120, 8),  # G5
                (center[1] + 185, center[0] - 120, 8),  # B5
                (center[1] - 260, center[0] - 120, 8),  # H5
                (center[1] + 260, center[0] - 120, 8),  # A5
                # Row 4
                (center[1] - 35, center[0] - 200, 10),  # E4
                (center[1] + 35, center[0] - 200, 10),  # D4
                (center[1] - 110, center[0] - 200, 10), # F4
                (center[1] + 110, center[0] - 200, 10), # C4
                (center[1] - 185, center[0] - 200, 8),  # G4
                (center[1] + 185, center[0] - 200, 8),  # B4
                (center[1] - 260, center[0] - 200, 8),  # H4
                (center[1] + 260, center[0] - 200, 8),  # A4
                # Row 3
                (center[1] - 35, center[0] - 280, 10),  # E3
                (center[1] + 35, center[0] - 280, 10),  # D3
                (center[1] - 110, center[0] - 280, 10), # F3
                (center[1] + 110, center[0] - 280, 10), # C3
                (center[1] - 185, center[0] - 280, 8),  # G3
                (center[1] + 185, center[0] - 280, 8),  # B3
                (center[1] - 260, center[0] - 280, 8),  # H3
                (center[1] + 260, center[0] - 280, 8),  # A3
                # Row 2
                (center[1] - 35, center[0] - 350, 9),   # E2
                (center[1] + 35, center[0] - 350, 9),   # D2
                (center[1] - 110, center[0] - 350, 9),  # F2
                (center[1] + 110, center[0] - 350, 9),  # C2
                (center[1] - 185, center[0] - 350, 8),  # G2
                (center[1] + 185, center[0] - 350, 8),  # B2
                (center[1] - 260, center[0] - 350, 8),  # H2
                (center[1] + 260, center[0] - 350, 8),  # A2
                # Row 1
                (center[1] - 35, center[0] - 415, 9),   # E1
                (center[1] + 35, center[0] - 415, 9),   # D1
                (center[1] - 110, center[0] - 415, 9),  # F1
                (center[1] + 110, center[0] - 415, 9),  # C1
                (center[1] - 185, center[0] - 415, 8),  # G1
                (center[1] + 185, center[0] - 415, 8),  # B1
                (center[1] - 260, center[0] - 415, 6),  # H1
                (center[1] + 250, center[0] - 415, 6),  # A1
                # Row 8
                (center[1] - 35, center[0] + 100, 10),  # E8
                (center[1] + 35, center[0] + 100, 10),  # D8
                (center[1] - 110, center[0] + 100, 10), # F8
                (center[1] + 110, center[0] + 100, 10), # C8
                (center[1] - 185, center[0] + 100, 8),  # G8
                (center[1] + 185, center[0] + 100, 8),  # B8
                (center[1] - 260, center[0] + 100, 8),  # H8
                (center[1] + 260, center[0] + 100, 8),  # A8
                # Row 9
                (center[1] - 35, center[0] + 175, 10),  # E9
                (center[1] + 35, center[0] + 175, 10),  # D9
                (center[1] - 110, center[0] + 175, 10), # F9
                (center[1] + 110, center[0] + 175, 10), # C9
                (center[1] - 185, center[0] + 175, 8),  # G9
                (center[1] + 185, center[0] + 175, 8),  # B9
                (center[1] - 260, center[0] + 175, 8),  # H9
                (center[1] + 260, center[0] + 175, 8),  # A9
                # Row 10
                (center[1] - 35, center[0] + 250, 10),  # E10
                (center[1] + 35, center[0] + 250, 10),  # D10
                (center[1] - 110, center[0] + 250, 10), # F10
                (center[1] + 110, center[0] + 250, 10), # C10
                (center[1] - 185, center[0] + 250, 8),  # G10
                (center[1] + 190, center[0] + 250, 8),  # B10
                (center[1] - 265, center[0] + 250, 8),  # H10
                (center[1] + 265, center[0] + 250, 8),  # A10
                # Row 11
                (center[1] - 35, center[0] + 330, 9),   # E11
                (center[1] + 35, center[0] + 330, 9),   # D11
                (center[1] - 110, center[0] + 330, 9),  # F11
                (center[1] + 110, center[0] + 330, 9),  # C11
                (center[1] - 185, center[0] + 330, 8),  # G11
                (center[1] + 190, center[0] + 330, 8),  # B11
                (center[1] - 265, center[0] + 330, 8),  # H11
                (center[1] + 265, center[0] + 330, 8),  # A11
                # Row 12
                (center[1] - 35, center[0] + 410, 9),   # E12
                (center[1] + 35, center[0] + 410, 9),   # D12
                (center[1] - 110, center[0] + 410, 9),  # F12
                (center[1] + 110, center[0] + 410, 9),  # C12
                (center[1] - 185, center[0] + 410, 8),  # G12
                (center[1] + 190, center[0] + 410, 8),  # B12
                (center[1] - 265, center[0] + 410, 6),  # H12
                (center[1] + 265, center[0] + 410, 6),  # A12
                ]


    # () -> New image with important pixels highlighted
    # changes the pixel color of centers of optimal parts of vials and color
    # of four pixels around them based on radius for testing of get_vials
    def draw_vials(self):
        vials = self.get_vials()
        image = self.image.copy()
        for vial in vials:
            # highlight pixel at center
            image[vial[1]][vial[0]] = [0, 0, 255]

            # also turn its neighboring pixels red for better visualization
            image[vial[1]+1][vial[0]+1] = [0, 0, 255]
            image[vial[1]+1][vial[0]] = [0, 0, 255]
            image[vial[1]+1][vial[0]-1] = [0, 0, 255]
            image[vial[1]][vial[0]+1] = [0, 0, 255]
            image[vial[1]][vial[0]-1] = [0, 0, 255]
            image[vial[1]-1][vial[0]+1] = [0, 0, 255]
            image[vial[1]-1][vial[0]] = [0, 0, 255]
            image[vial[1]-1][vial[0]-1] = [0, 0, 255]

            # draw the corners of the square based on the radius in green
            # highlight their neighbors are well for better visualization
            top = vial[1]-vial[2]
            bottom = vial[1]+vial[2]
            left = vial[0]-vial[2]
            right = vial[0]+vial[2]

            # top left
            image[top][left] = [0, 255, 0]
            image[top+1][left+1] = [0, 255, 0]
            image[top+1][left] = [0, 255, 0]
            image[top+1][left-1] = [0, 255, 0]
            image[top][left] = [0, 255, 0]
            image[top][left] = [0, 255, 0]
            image[top-1][left+1] = [0, 255, 0]
            image[top-1][left] = [0, 255, 0]
            image[top-1][left-1] = [0, 255, 0]

            # top right
            image[top][right] = [0, 255, 0]
            image[top+1][right+1] = [0, 255, 0]
            image[top+1][right] = [0, 255, 0]
            image[top+1][right-1] = [0, 255, 0]
            image[top][right] = [0, 255, 0]
            image[top][right] = [0, 255, 0]
            image[top-1][right+1] = [0, 255, 0]
            image[top-1][right] = [0, 255, 0]
            image[top-1][right-1] = [0, 255, 0]

            # bottom left
            image[bottom][left] = [0, 255, 0]
            image[bottom+1][left+1] = [0, 255, 0]
            image[bottom+1][left] = [0, 255, 0]
            image[bottom+1][left-1] = [0, 255, 0]
            image[bottom][left] = [0, 255, 0]
            image[bottom][left] = [0, 255, 0]
            image[bottom-1][left+1] = [0, 255, 0]
            image[bottom-1][left] = [0, 255, 0]
            image[bottom-1][left-1] = [0, 255, 0]

            # bottom right
            image[bottom][right] = [0, 255, 0]
            image[bottom+1][right+1] = [0, 255, 0]
            image[bottom+1][right] = [0, 255, 0]
            image[bottom+1][right-1] = [0, 255, 0]
            image[bottom][right] = [0, 255, 0]
            image[bottom][right] = [0, 255, 0]
            image[bottom-1][right+1] = [0, 255, 0]
            image[bottom-1][right] = [0, 255, 0]
            image[bottom-1][right-1] = [0, 255, 0]

        cv2.imshow("Vial locations", imutils.resize(image, width=500))

    # () -> [[color]]
    # reads the colors of each vial on the plate
    # returns some data structure to demonstrate the array
    def get_colors(self):
        raise "not yet implemented"
    # TODO: maybe remove

    # (maybe path) -> () (saves image)
    # read the colors of the image and export image to paht
    def export_colors(self, path="."):
        raise "not yet implemented"
    # TODO: maybe remove

   
    # () -> New cropped image
    # normalize the image to have a rectangular shape
    def normalize_shape(self):
        rect = self.detect_bounds()
        image = self.image.copy()

        # rotate img
        angle = rect[2]
        rows, cols = image.shape[0], image.shape[1]
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_rot = cv2.warpAffine(image, m, (cols, rows))

        # rotate bounding box
        box = cv2.boxPoints(rect)
        pts = np.int0(cv2.transform(np.array([box]), m))[0]
        pts[pts < 0] = 0

        # crop
        img_crop = img_rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]

        return PlateImage(img_crop)


    # calculates the dominant RGB value of a specified area on an image
    # @param a tuple with x coord, y coord, and radius at (x,y)  on image
    # -- (x, y, radius)
    # @return list with the RGB values and location parameters as tuples
    # -- ((r, g, b), (x, y, radius))
    def find_color(self, location):

        image = self.image.copy()
        x, y, radius = location
        cropped = image[(x - radius):(x + radius), (y - radius):(y + radius)]
        cv2.imshow("Cropped", cropped)
        cv2.waitKey(0)

        NUM_CLUSTERS = 5
        ar = np.asarray(cropped)
        shape = ar.shape
        ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

        # assign codes and count occurrences
        vecs, dist = scipy.cluster.vq.vq(ar, codes)
        counts, bins = np.histogram(vecs, len(codes))

        # find most frequent
        index_max = np.argmax(counts)
        dominant_color = codes[index_max]

        # formats to RGB tuple by rounding to int, reversing the list,
        # and converting to tuple
        dominant_color = tuple(
            [int(round(num)) for num in dominant_color][::-1])

        return [dominant_color, location]

    # () -> () (mutates object)
    # normalize the colors of the image to improve color accuracy
    def normalize_color(self):
        raise "not yet implemented"

    # (maybe path) -> ()
    # writes the image to disk at provided path
    def save(self, path="."):
        cv2.imwrite(path, self.image)

    # displays image through opencv
    # () -> ()
    def show(self):
        cv2.imshow("ELISA Plate", self.image)
