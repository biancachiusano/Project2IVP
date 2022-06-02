import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise


# 1.1
img = cv2.imread('images project 2/bird.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.double)

w = img.shape[0]
h = img.shape[1]

[u,v] = np.mgrid[-w/2:w/2, -h/2:h/2]
u = 2*u/w
v = 2*v/h

F = np.fft.fft2(img)

a = 0.2
b = 0.2

H = np.sinc((u*a + v*b)) * np.exp(-1j*np.pi*(u*a + v*b))
G = F
G[:,:,2] = F[:,:,2] * H
G[:,:,1] = F[:,:,1] * H
G[:,:,0] = F[:,:,0] * H

g = np.fft.ifft2(G)
plt.imshow(np.abs(g)/255)
plt.show()

# Gaussian Noise
noise = random_noise(np.abs(g).astype(np.uint8), 'gaussian', mean=0.2, var=0.02)
noise = noise.astype(np.double)
plt.imshow(noise)
plt.show()
