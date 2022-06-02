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

def computDCT(blockSize):
    # Do 8x8 DCT on image (in-place)
    for i in r_[:imsize[0]:blockSize]:
        for j in r_[:imsize[1]:blockSize]:
            dct[i:(i+blockSize),j:(j+blockSize)] = dct2( im[i:(i+blockSize),j:(j+blockSize)] )
    print(dct.shape)
    pos = 128
    # Extract a block from image
    plt.figure()
    plt.imshow(im[pos:pos+blockSize,pos:pos+blockSize],cmap='gray')
    plt.title("An {} by {} Image block".format(blockSize, blockSize))

    # Display the dct of that block
    plt.figure()
    plt.imshow(dct[pos:pos+blockSize,pos:pos+blockSize],cmap='gray',vmax= np.max(dct)*0.01,vmin = 0, extent=[0,pi,pi,0])
    plt.title("An {} by {} DCT block".format(blockSize, blockSize))

computDCT(8)
# Display entire DCT
plt.figure()
plt.imshow(dct,vmax = np.max(dct)*0.01,vmin = 0)
plt.title( "8x8 DCTs of the image")

# Find the inverse dct for various values of K (until you choose one)
# at this point there are 64 DCT coefficients
# order dct coefficients by abs magnitude
# THIS NEEDS TO BE DONE FOR EACH BLOCK
dct_size = dct.shape
dct_abs = np.zeros(dct_size)
ordered = np.zeros(dct_size)
print(dct)
for c in range(64):
    dct_abs[c] = abs(dct[c])
print(dct_abs)
ordered = np.sort(dct_abs)[::-1]
print(ordered)

k = 32

'''
# Threshold
thresh = 0.012
dct_thresh = dct * (abs(dct) > (thresh*np.max(dct)))

plt.figure()
plt.imshow(dct_thresh,vmax = np.max(dct)*0.01,vmin = 0)
plt.title( "Thresholded 8x8 DCTs of the image")

percent_nonzeros = np.sum( dct_thresh != 0.0 ) / (imsize[0]*imsize[1]*1.0)
print ("Keeping only %f%% of the DCT coefficients" % (percent_nonzeros*100.0))

# Compare to original
img_dct = np.zeros(imsize)
for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        img_dct[i:(i + 8), j:(j + 8)] = idct2(dct_thresh[i:(i + 8), j:(j + 8)])

plt.figure()
plt.imshow(np.hstack((im, img_dct)), cmap='gray')
plt.title("Comparison between original and DCT compressed images")

'''

plt.show()