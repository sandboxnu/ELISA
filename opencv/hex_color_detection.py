# import numpy
import cv2

#intakes x and y coordinates
x = int(input("Enter an integer for the x coordinate: "))
y = int(input("Enter an integer for the y coordinate: "))

print (x)
print (y)

# opens image file
image = cv2.imread("../images/1.jpeg")
image = cv2.resize(image, (340,480))

# shows image and waits til key is pressed to close
cv2.imshow("Image", image)
#cv2.waitKey(0)
cv2.destroyAllWindows()

# prints color as tuple
color = image[y, x]
print (color)

# prints hexColor
hexColor = (color[0] << 16) + (color[1] << 8) + (color[2])
print("Hex color: " + str(hex(hexColor)))