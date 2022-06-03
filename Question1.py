import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

def blur_image(channel, a, b ):
    w = channel.shape[0]
    h = channel.shape[1]

    [u, v] = np.mgrid[-w / 2:w / 2, -h / 2:h / 2]
    u = 2 * u / w
    v = 2 * v / h

    F = np.fft.fft2(channel)
    H = np.sinc((u * a + v * b)) * np.exp(-1j * np.pi * (u * a + v * b))

    G = F*H
    g = np.fft.ifft2(G)
    channel_blur = np.abs(g) / 255
    return channel_blur


def inverse_blur(channel, a, b):
    w = channel.shape[0]
    h = channel.shape[1]

    [u, v] = np.mgrid[-w / 2:w / 2, -h / 2:h / 2]
    u = 2 * u / w
    v = 2 * v / h

    F = np.fft.fft2(channel)
    H = np.sinc((u * a + v * b)) * np.exp(-1j * np.pi * (u * a + v * b))

    G = F / H
    g = np.fft.ifft2(G)
    channel_inv = np.abs(g) /255
    return channel_inv

def gaussian_noise(img):
    noise = random_noise(img, 'gaussian', mean=0, var=0.02)
    return noise

def wiener_filtering(o, n):
    o_ps = np.abs(o)**2
    n_ps = np.abs(n)**2
    #k = np.sum(n_ps)/np.sum(o_ps)
    k = 1
    combined = np.fft.fftshift(np.fft.fft2(o+n))
    back_fft = combined/(1 + ((n_ps/o_ps)*k))
    return np.abs(np.fft.ifft2(back_fft))


img = cv2.imread('images project 2/bird.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.double)

# blur
(ch1, ch2, ch3) = cv2.split(img)
ch1_b = blur_image(ch1, 0, 0.2)
ch2_b = blur_image(ch2, 0, 0.2)
ch3_b = blur_image(ch3, 0, 0.2)
blurred = cv2.merge((ch1_b, ch2_b, ch3_b))
blurred = blurred/np.max(blurred)
plt.imshow(blurred)
plt.show()
# gaussian noise
g_noise = gaussian_noise(blurred)
plt.imshow(g_noise)
plt.show()

# inverse blur
(ch1_br, ch2_br, ch3_br) = cv2.split(blurred)
ch1_i = inverse_blur(ch1_br, 0, 0.2)
ch2_i = inverse_blur(ch2_br, 0, 0.2)
ch3_i = inverse_blur(ch3_br, 0, 0.2)
inverse = cv2.merge((ch1_i, ch2_i, ch3_i))
inverse = inverse/np.max(inverse)
plt.imshow(inverse)
plt.show()

# inverse blur and noise
(ch1_n, ch2_n, ch3_n) = cv2.split(g_noise)
ch1_inb = inverse_blur(ch1_n, 0, 0.2)
ch2_inb = inverse_blur(ch2_n, 0, 0.2)
ch3_inb = inverse_blur(ch3_n, 0, 0.2)
inverse_noise = cv2.merge((ch1_inb, ch2_inb, ch3_inb))
inverse_noise = inverse_noise/np.max(inverse_noise)
plt.imshow(inverse_noise)
plt.show()

def mmse(original_c, degraded_c, h):
    o = np.fft.fftshift(np.fft.fft2(original_c))
    s_f = np.abs(o) ** 2
    s_n = np.abs(np.fft.fftshift(np.fft.fft2(degraded_c))) ** 2
    denominator = np.abs(h)**2 + (s_n/s_f)
    Hw = np.conj(h) / denominator
    Fhat = Hw * o
    return Fhat


(o_ch1, o_ch2, o_ch3) = cv2.split(img)
(noise_ch1, noise_ch2, noise_ch3) = cv2.split(g_noise)
Fhat_1 = mmse(o_ch1, noise_ch1, 1)
Fhat_2 = mmse(o_ch2, noise_ch2, 1)
Fhat_3 = mmse(o_ch3, noise_ch3, 1)

result1 = np.abs(np.fft.ifft2(Fhat_1))
result2 = np.abs(np.fft.ifft2(Fhat_2))
result3 = np.abs(np.fft.ifft2(Fhat_3))
result = cv2.merge((result1,result2,result3))
result = result/np.max(result)
plt.imshow(result)
plt.show()