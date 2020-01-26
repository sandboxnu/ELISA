import cv2

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        # contour approximation
        # second parameter is approximation accuracy
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) >=4:
            shape = "plate"
        else:
            shape = "unknown"

        return shape
