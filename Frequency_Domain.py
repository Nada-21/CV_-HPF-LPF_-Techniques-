import numpy as np 
import cv2
import skimage.io as io

#Function to resize image
def size(image):
    resized_image = cv2.resize(image,(400,300))
    return resized_image

#Convert RGB image to GrayScale image
def grayscale(path):
    color_img=io.imread(path)
    red = color_img[:, :, 0]
    green = color_img[:, :, 1]
    blue = color_img[:, :, 2]
    img_grey = 0.299 *red + 0.587 *green + 0.114 *blue
    # img_grey = np.dot(color_img[...,:3],[0.299,0.587,0.114])
    img_grey = size(img_grey)
    return img_grey

#Low Pass Filter
def lowPassFilter(image) :
    fftimg = np.fft.fft2(image)
    #shifting low frequncies to the center 
    fftshiftimg = np.fft.fftshift(fftimg)
    M,N = image.shape
    H = np.zeros((M,N), dtype=np.float32)
    D0 = 50
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            if D <= D0:
               H[u,v] = 1
            else:
               H[u,v] = 0       
    # Ideal Low Pass Filtering
    Gshift = fftshiftimg * H
    # Inverse Fourier Transform
    ifftimg = np.fft.ifftshift(Gshift)
    lpfilterdimg = np.abs(np.fft.ifft2(ifftimg))
    return lpfilterdimg, H, fftshiftimg

#High Pass Filter
def highPassFilter(image) :
    lpfilterdimg, H, fftshiftimg = lowPassFilter(image)
    H = 1 - H
    Gshift= fftshiftimg * H
    ifftimg = np.fft.ifftshift(Gshift)
    hpfilterdimg = np.abs(np.fft.ifft2(ifftimg))
    return hpfilterdimg

def lowpasscv2 (image):
    kernel = np.ones((10,10),np.float32)/25
    Lp = cv2.filter2D(image,-1, kernel)
    filterdimg = image - Lp
    
    return filterdimg 

def highpasscv2 (image):
    dst = cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)
    Hp = image - dst
    return Hp
