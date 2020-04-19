#!/usr/bin/env python3
import imutils
import cv2
import numpy as np
import pandas as pd
import scipy.cluster


# [('label, num)] -> ['label]
# determines whether there is an outlier in the data
def find_outliers(data, thresh=3):
    data = np.array(data)
    mean = np.mean(data)
    stddev = np.std(data)

    outliers = []

    for label, y in data:
        z = (y - mean) / stddev
        if np.abs(z) > thresh:
            outliers.append(label)

    return outliers

# creates a PlateImage given a path
def from_path(image_path):
    return PlateImage(cv2.imread(image_path))

# [('label, num)] -> bool
# determines whether there are outliers in the data
def has_outliers(data, thresh=3):
    return find_outliers(data, thresh=thresh) != []


# Represents an image of an ELISA plate
class PlateImage:

    # default constructor
    # creates image from provided image path
    # when adding parameters (such as thresholds),
    # these should be optional parameters for methods
    # ASSUME image is always scaled to have width of 700
    def __init__(self, img):
        self.image = imutils.resize(img, width=700)

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
        corner_offset = 25
        num_rows, num_cols, num_dims = image.shape
        top_rows   = range(corner_offset)
        bot_rows   = range((num_rows-corner_offset), num_rows)
        left_cols  = range(corner_offset)
        right_cols = range((num_cols-corner_offset), num_cols)

        # find the average color value of the pixels in a given corner of image
        # throw an error if they are lighter than the upperTresh
        def avg_color(rows, cols):
            total_red   = 0
            total_green = 0
            total_blue  = 0
            for r in rows:
                for c in cols:
                    total_red   += image[r, c][0]
                    total_green += image[r, c][1]
                    total_blue  += image[r, c][2]
            avg = np.array([(total_red/625),
                            (total_green/625),
                            (total_blue/625)])  # TODO magic numbers

            # check to see if each component of the rgb value is darker than
            # upper boundary, otherwise throw an error
            for avg_color, thresh in zip(avg, upper_thresh):
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

        row_adjustments = [-35, 35, -110, 110, -185, 185, -260, 260]
        col_adjustments = [28, -50, -120, -200, -280, -350,
                           -415, 100, 175, 250, 330, 410]
        radius = 10

        return [(center[1] + r_adj, center[0] + c_adj, radius)
                for r_adj in row_adjustments
                for c_adj in col_adjustments]


    # () -> New image with important pixels highlighted
    # changes the pixel color of centers of optimal parts of vials and color
    # of four pixels around them based on radius for testing of get_vials
    def draw_vials(self):
        vials = self.get_vials()
        image = self.image.copy()

        # image = imutils.resize(image, width=700)

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
            # highligh their neighbors are well for better visualization
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

        # RED = [0, 0, 255]
        # GREEN = [0, 255, 0]
        # pos_offsets = [-1, 0, 1]
#
        # for vial in vials:
            # # highlight pixel at center
            # image[vial[1]][vial[0]] = RED
#
            # for x_off in pos_offsets:
                # for y_off in pos_offsets:
                    # # also turn its neighboring pixels red
                    # # for better visualization
                    # image[vial[1]+x_off][vial[0]+y_off] = RED
#
            # # draw the corners of the square based on the radius in green
            # # highlight their neighbors are well for better visualization
            # top    = vial[1]-vial[2]
            # bottom = vial[1]+vial[2]
            # left   = vial[0]-vial[2]
            # right  = vial[0]+vial[2]
#
            # # locations of the four corners
            # corner_locs = [(top, left),
                           # (top, right),
                           # (bottom, left),
                           # (bottom, right)]
#
            # # top left
            # for x_loc, y_loc in corner_locs:
                # for x_off in pos_offsets:
                    # for y_off in pos_offsets:
                        # image[x_loc + x_off][y_loc + y_off] = GREEN
#
        # cv2.imshow("Vial locations", imutils.resize(image, width=700))


    # () -> [((r, g, b), (x, y, radius))]
    # reads the colors of each vial on the plate
    # returns some data structure to represent colors on array
    def get_colors(self):
        try:
            if self.is_blurry():
                raise ValueError("Image was blurry!")
            # self.check_background()
            img = self.normalize_shape()  # 500w image
            return [img.find_color(loc) for loc in img.get_vials()]  # [()]

        except(ValueError):
            raise ValueError("The image could not be read from. Please try again!")


    # (maybe path) -> PlateImage
    # read the colors of the image and export image to paht
    def export_colors(self, path="."):
        try:
            return self.draw_img_data(self.get_colors())
        except(ValueError):
            raise ValueError("The image could not be read from. Please try again.")


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

        # crop the image based on the corners found
        img_crop = img_rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]

        return PlateImage(img_crop)


    # calculates the dominant RGB value of a specified area on an image
    # @param a tuple with x coord, y coord, and radius at (x,y)  on image
    # -- (x, y, radius)
    # @return tuple with the RGB values and location parameters as tuples
    # -- ((r, g, b), (x, y, radius))
    def find_color(self, location):
        print("finding the color...")

        image = imutils.resize(self.image.copy(), 710)
        x, y, radius = location
        cropped = image[(x - radius):(x + radius), (y - radius):(y + radius)]

        print(x)
        print(y)
        print(radius)
        print(cropped)

        NUM_CLUSTERS = 5
        ar = np.asarray(cropped)
        shape = ar.shape
        ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

        # TODO: bug is here
        print(ar)
        print(NUM_CLUSTERS)

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

        return (dominant_color, location)

    # () -> () (mutates object)
    # normalize the colors of the image to improve color accuracy
    def normalize_color(self):
        raise "not yet implemented"
        # TODO: unclear whether we need to do this

    # (maybe path) -> ()
    # writes the image to disk at provided path
    def save(self, path="./"):
        cv2.imwrite(path, self.image)

    # displays image through opencv
    # () -> ()
    def show(self):
        cv2.imshow("ELISA Plate", self.image)

    # converts a list of relative colors to a list of abs colors
    # [(r,g,b)] -> [0...1]
    def rel_to_abs_color(self, color_list):
        abs_list = [r + g + b for r, g, b in color_list]
        return [float(val)/sum(abs_list) for val in abs_list]

    # draw rgb data on the image
    # () -> PlateImage
    # throws Exception if the image data is not realistic
    def draw_img_data(self, img_data):

        # image display configuration
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.2       # scale of the font
        color = (0, 0, 255)  # red
        thickness = 1        # line thickness

        img = self.image.copy()

        # obtain relative colors list
        # ASSUME rel colors list is same length as abs colors list
        rel_list = self.rel_to_abs_color([rgb for (rgb, pos) in img_data])

        # if has_outliers(rel_list):
            # raise ValueError("A color reading was too extreme to be realistic.")

        for ((r, g, b), (x, y, rad)), rel in zip(img_data, rel_list):
            printstr = "({:07.3f}, {:07.3f}, {:07.3f}):{:07.3f}".format(r, g, b, rel)
            print(printstr)

            cv2.putText(
                img,
                # prints (r, g, b):rel
                printstr,
                (x, y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )

        return PlateImage(img)
