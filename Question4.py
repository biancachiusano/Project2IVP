import glob

import numpy as np
import cv2

def reshapeImages(array):
    reshapedPictures = []
    for i in range(0, len(array)):
        img = cv2.resize(array[0], (4000, 6000), interpolation=cv2.INTER_LINEAR)
        reshapedPictures.append(img)
    return reshapedPictures

def dataMatrix(images):
    length = len(images)
    overallSize = images[0].shape
    data = np.zeros((length, overallSize[0]*overallSize[1]*overallSize[2]), dtype=np.uint8)
    for i in range(0, length):
        image = images[i].flatten()
        data[i,:] = image
    return data

def vectorsAndMean(mat):
    return cv2.PCACompute(mat, mean=None)

def getEigenFaces(vector):
    eigenFaces = []
    for vec in eigenVectors:
        eigenFaces.append(vec)
    return eigenFaces

def calculateWeights(mat, vectors, mean):
    weights = []
    for i in range(mat.shape[0]):
        img = np.squeeze(np.asarray(mat[i]), axis=0)
        m = img - mean
        weight = np.dot(m,vectors[i])
        weights.append(weight)
    return weights

menPictures = []
for img in glob.glob("images project 2/man*.jpg"):
    menPictures.append(cv2.imread(img))
r_men = reshapeImages(menPictures)

overall_size = r_men[0].shape
mat = dataMatrix(r_men)
mean, eigenVectors = vectorsAndMean(mat)
averageFace = mean.reshape(overall_size)
eigenFaces = getEigenFaces(eigenVectors)
weights = calculateWeights(mat, eigenVectors, mean)
output = averageFace

for i in range(0, len(eigenFaces)):
    output = np.add(output, eigenFaces[i]*weights[i])
output.astype(np.uint8).reshape((6000,4000,3))

cv2.imshow("output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
oldPictures = []
for img in glob.glob("images project 2/old*.jpg"):
    oldPictures.append(cv2.imread(img))
r_old = reshapeImages(oldPictures)

womanPictures = []
for img in glob.glob("images project 2/woman*.jpg"):
    womanPictures.append(cv2.imread(img))
r_woman = reshapeImages(womanPictures)
'''


