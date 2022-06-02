import numpy as np
import matplotlib.pyplot as plt
import scipy

import scipy.fftpack
from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
import imageio.v2 as imageio
import matplotlib.pylab as pylab

# https://codetobuy.com/downloads/discrete-cosine-transform-dct-on-each-8x8-block-of-the-image/
# https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html

pylab.rcParams['figure.figsize'] = (20.0, 7.0)

im = imageio.imread("images project 2/cameraman.tif").astype(float)
f = plt.figure()
plt.imshow(im,cmap='gray')

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


imsize = im.shape
dct = np.zeros(imsize)

# Do 8x8 DCT on image (in-place)
for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        dct[i:(i+8),j:(j+8)] = dct2( im[i:(i+8),j:(j+8)] )

pos = 128

# Extract a block from image
plt.figure()
plt.imshow(im[pos:pos+8,pos:pos+8],cmap='gray')
plt.title( "An 8x8 Image block")

# Display the dct of that block
plt.figure()
plt.imshow(dct[pos:pos+8,pos:pos+8],cmap='gray',vmax= np.max(dct)*0.01,vmin = 0, extent=[0,pi,pi,0])
plt.title( "An 8x8 DCT block")

# Display entire DCT
plt.figure()
plt.imshow(dct,vmax = np.max(dct)*0.01,vmin = 0)
plt.title( "8x8 DCTs of the image")

plt.show()