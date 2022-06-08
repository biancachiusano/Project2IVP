import numpy as np
import matplotlib.pyplot as plt
import scipy

import scipy.fftpack
from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from skimage import color
from skimage import io
import imageio.v2 as imageio
import matplotlib.pylab as pylab

# https://codetobuy.com/downloads/discrete-cosine-transform-dct-on-each-8x8-block-of-the-image/
# https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
# https://nl.mathworks.com/help/images/discrete-cosine-transform.html
# https://en.wikipedia.org/wiki/Discrete_cosine_transform

# Code from looking at the lab

pylab.rcParams['figure.figsize'] = (20.0, 7.0)

im = imageio.imread("images project 2/geese.jpg")
im = color.rgb2gray(im)
im = im.astype(float)
f = plt.figure()
plt.imshow(im,cmap='gray')
imsize = im.shape
dct = np.zeros(imsize)

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')
# Inverse DCT
def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.T , norm='ortho').T,norm='ortho')

def computeDCT(blockSize):
    # Do 8x8 DCT on image (in-place)
    for i in r_[:imsize[0]:blockSize]:
        for j in r_[:imsize[1]:blockSize]:
            dct[i:(i+blockSize),j:(j+blockSize)] = dct2( im[i:(i+blockSize),j:(j+blockSize)] )
    return dct


def computeInverse(dct, blockSize):
    new_img = np.zeros(dct.shape)
    for i in r_[:new_img.shape[0]:blockSize]:
        for j in r_[:new_img[1]:blockSize]:
            new_img[i:(i+blockSize),j:(j+blockSize)] = idct2( dct[i:(i+blockSize),j:(j+blockSize)] )
    return new_img

blockSize = 8
dct = computeDCT(blockSize)

pos = 128
# Extract a block from image
plt.figure()
plt.imshow(im[pos:pos+blockSize,pos:pos+blockSize],cmap='gray')
plt.title("An {} by {} Image block".format(blockSize, blockSize))

# Display the dct of that block
plt.figure()
plt.imshow(dct[pos:pos+blockSize,pos:pos+blockSize],cmap='gray',vmax= np.max(dct)*0.01,vmin = 0, extent=[0,pi,pi,0])
plt.title("An {} by {} DCT block".format(blockSize, blockSize))

# Display entire DCT
plt.figure()
plt.imshow(dct,vmax = np.max(dct)*0.01,vmin = 0)
plt.title( "8x8 DCTs of the image")

thresh = 0.012
dct_thresh = dct * (abs(dct) > (thresh * np.max(dct)))


plt.show()