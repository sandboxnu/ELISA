import numpy
import cv2

#intakes x and y coordinates
x = input("Enter x coordinate: ")
y = input("Enter y coordinate: ")

print (x)
print (y)

# opens image file
image = cv2.imread("../images/IMG_0675.jpg")
image = cv2.resize(image, (340,480))

# shows image and waits til key is pressed to close
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# prints color as tuple
for x in range (0,340,1):
    for y in range(0,480,1):
        color = image[y,x]
        print (color)
# color = image[y,x]

# prints hexColor
hexColor = (color[0] << 16) + (color[1] << 8) + (color[2])
print("Hex color: " + str(hex(hexColor)))