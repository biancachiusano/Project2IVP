import cv2
from skimage.measure import label
from skimage.measure import regionprops

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


#Convert Image
oranges = cv2.imread("images project 2/oranges.jpg", 0)
orangeTree = cv2.imread("images project 2/orangetree.jpg", 0)

# https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-1-simple-thresholding/#:~:text=Thresholding%20is%20a%20technique%20in,maximum%20value%20(generally%20255).
ret, thresh1Oranges = cv2.threshold(oranges, 120, 255, cv2.THRESH_BINARY) # 120 is the threshold
ret, thresh1OrangeTree = cv2.threshold(orangeTree, 120, 255, cv2.THRESH_BINARY)

cv2.imwrite("outputImages/orangesThresh.jpg", thresh1Oranges)
cv2.imwrite("outputImages/orangeTreeThresh.jpg", thresh1OrangeTree)

# Lecture 7 open and close
# https://www.javatpoint.com/opencv-erosion-and-dilation#:~:text=Erosion%20and%20Dilation%20are%20morphological,or%20structure%20of%20an%20object.
# kernel = np.ones((6,6), np.uint8)
img1_count = countOranges(thresh1OrangeTree)
print(img1_count)

#cv2.imwrite("outputImages/morphImageProcessing.jpg", dilateOTree)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
