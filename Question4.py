import glob
import numpy as np
import cv2

def createDataMatrix(pictures):
    mat = np.stack(pictures, axis=0)
    mat = mat / 255
    return mat

def pca(mat):
    return cv2.PCACompute(mat, mean=None)


womanPictures = []
for img in glob.glob("images project 2/PCAImages/woman*.JPG"):
    womanPictures.append(cv2.imread(img))

h, w, c = womanPictures[0].shape
data_matrix = createDataMatrix(womanPictures)
mean = np.mean(data_matrix, axis=0)
data_matrix = np.subtract(data_matrix, mean)
m, eigen_vectors = pca(data_matrix)
mean_face = np.reshape(m, (h, w, c))

manPictures = []
for img in glob.glob("images project 2/PCAImages/man4*.JPG"):
    manPictures.append(cv2.imread(img))


#mean, eigen_vectors  = pca(data_matrix)
#eigen_faces = [np.reshape(vec, (h, w, c)) for vec in eigen_vectors]
#print("There are {} eigen faces".format(len(eigen_faces)))

cv2.imshow("face1", mean_face)
cv2.waitKey(0)
cv2.destroyAllWindows()