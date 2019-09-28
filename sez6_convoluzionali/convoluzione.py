import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
import numpy as np

def showimg(img):
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.figure(figsize=(12,12))
    plt.show()

img = io.imread("/home/daniaffch/Scrivania/AI_t/sez6_convoluzionali/django.jpg")
img_bw = rgb2gray(img)
showimg(img_bw)

#convoluzione
from scipy.signal import convolve2d
#linee verticali
filter = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
feature_map = convolve2d(img_bw, filter, mode='full', boundary='fill', fillvalue=0)
showimg(feature_map)

#linee orizzontali
filter = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
feature_map = convolve2d(img_bw, filter, mode='full', boundary='fill', fillvalue=0)
showimg(feature_map)
#edge
filter = np.array([[0,1,0],[1,-4,1],[0,1,0]])
feature_map = convolve2d(img_bw, filter, mode='full', boundary='fill', fillvalue=0)
showimg(feature_map)


#IMMAGINI A COLORI
#immagini 3d (rgb)
from cv2 import filter2D
#edge
filter = np.array([[0,1,0],[1,-4,1],[0,1,0]])
feature_map = filter2D(img, -1, filter)
showimg(feature_map)
