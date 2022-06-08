import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.measure import regionprops
# https://www.javatpoint.com/opencv-erosion-and-dilation#:~:text=Erosion%20and%20Dilation%20are%20morphological,or%20structure%20of%20an%20object.

def closing(img, k):
    dilate = cv2.dilate(img, k)
    erode = cv2.erode(dilate, k)
    return erode


def opening(img, k):
    erode = cv2.erode(img, k)
    dilate = cv2.dilate(erode, k)
    return dilate

def countOranges(img):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    open = opening(img, k)
    close = closing(open, k)
    label_img = label(close)
    regions = regionprops(label_img)
    count = 0
    for num, x in enumerate(regions):
        area = x.area
        convex_area = x.convex_area
        if (num!=0 and (area>10) and (convex_area/area < 1.05) and (convex_area/area > 0.95)):
            count = count + 1

    return count

# Granulometry
def granulometry(img, size):
    areas = np.zeros(len(size))

    for i in range(len(size)):
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i+1, i+1))
        open = opening(img, se)
        areas[i] = np.sum(open)
    return open, areas

def granulometry_two(img, size):
    s = 1
    s_open = img
    iter = size
    while iter > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s,s))
        open = opening(s_open, se)
        s = s + 1
        s_open = open
        iter = iter - 1
    return open

# Task 3.1
#Convert Image
oranges = cv2.imread("images project 2/oranges.jpg", 0)
orangeTree = cv2.imread("images project 2/orangetree.jpg", 0)

# https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-1-simple-thresholding/#:~:text=Thresholding%20is%20a%20technique%20in,maximum%20value%20(generally%20255).
ret, thresh1Oranges = cv2.threshold(oranges, 120, 255, cv2.THRESH_BINARY) # 120 is the threshold
ret, thresh1OrangeTree = cv2.threshold(orangeTree, 120, 255, cv2.THRESH_BINARY)

img1_count = countOranges(thresh1OrangeTree)
print(img1_count)


# Task 3.2
# Load picture
jar = cv2.cvtColor(cv2.imread("images project 2/jar.jpg"), cv2.COLOR_BGR2RGB)
plt.imshow(jar)
plt.show()

# Convert to grayscale - preprocessing
gray_jar = cv2.cvtColor(jar, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_jar, cmap='gray')
plt.show()

# Apply grayscale morphological operations
# size = np.linspace(1, 100, 100)
# gran, frequencies = granulometry(gray_jar, size)
gran = granulometry_two(gray_jar, 30)
plt.imshow(gran, cmap='gray')
plt.show()

#plt.plot(size, frequencies, '-bo')
#plt.title('Difference in surface area, with different radius')
#plt.xlabel('Size')
#plt.ylabel('Differences in surface area')
#plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
