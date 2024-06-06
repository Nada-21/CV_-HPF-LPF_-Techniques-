import numpy as np
from scipy import ndimage

# ----------------------------------------------Noise-----------------------------------------------------------
# add uniform noise
def Uniform_noise(image):
    img=image
    img=img/255          #normalization 
    # uniform noise
    x,y=img.shape                                                     # uniform distribution p(z)=1/(b-a) if a<=z<=b
    a=0                                                               #                       p(z)=0      otherwise 
    b=0.5
    noise=np.random.uniform(size = (x,y), low = a, high = b)
    # add noise to image
    noise_img = img + noise   
    return noise_img

# add gaussian noise
def Gaussian_noise(image):
    img=image
    img=img/255          #normalization
    # create gaussion noise                                    Gaussian Noise Formula:  
    x,y=img.shape                                               #   p(z)=(1/sigma*sqrt(2pi))*e**(- (z-mean)**2/2*var)                 
    mean = 0
    var = 0.01
    sigma = np.sqrt(var)
    noise= np.random.normal(loc=mean,scale=sigma,size=(x,y))
    # add a gaussian noise
    noise_img = img + noise
    return noise_img

# add salt and pepper noise
def salt_pepper_noise(image):
    # salt=white=1   pepper=black=0
    img=image
    img=img/255          #normalization
    # black image 
    x,y = img.shape
    noise=np.zeros((x,y), dtype=np.float64)
    # salt and pepper amount
    pepper = 0.1
    salt = 1 - pepper
    #create salt and pepper noise image
    for i in range(x):
        for j in range(y):
            rdn = np.random.random()
            if rdn < pepper:
                noise[i][j] = 0
            elif rdn > salt:
                noise[i][j] = 1
            else:
                noise[i][j] = img[i][j]
    
    return noise

# --------------------------------------------------Filters----------------------------------------------------------

def average_filter(image,kernel_size):
    img=image
    # Obtain number of rows and columns of the image
    x, y = img.shape
    # Develop Averaging filter(3, 3) mask
    kernel = np.ones([kernel_size , kernel_size ], dtype = int)
    kernel = kernel / (kernel_size**2)
    Xnew=x-kernel_size +1
    Ynew=y-kernel_size +1
    # Convolve the 3X3 mask over the image 
    new_image = np.zeros([Xnew , Ynew])
    for i in range(Xnew):
        for j in range(Ynew):
            value=np.multiply(img[i:i+kernel_size,j:j+kernel_size], kernel)
            new_image[i][j]=np.sum(value)
    return new_image


def Gaussian_filter(image,kernel_size,sigma):
    img=image
    # Obtain number of rows and columns of the image
    m, n = img.shape
    # Develop gaussian filter(3, 3) mask
    x, y = np.meshgrid(np.linspace(-1,1,kernel_size ), np.linspace(-1,1,kernel_size ))
    d = -(x*x+y*y)
    kernel= np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )/(2*np.pi*sigma**2)
    Xnew=m-kernel_size +1
    Ynew=n-kernel_size +1

    # Convolve the 3X3 mask over the image 
    img_new = np.zeros([Xnew , Ynew ])
    for i in range(Xnew):
        for j in range(Ynew):
            value=np.multiply(img[i:i+kernel_size,j:j+kernel_size], kernel)
            img_new[i][j]=np.sum(value)
    return img_new


def median_filter(image,size):
    img=image
    # Obtain number of rows and columns of the image
    x, y = img.shape
    Xnew=x-size +1
    Ynew=y-size +1
    # Traverse the image. For every 3X3 area, 
    # find the median of the pixels and
    # replace the center pixel by the median
    img_new = np.zeros([Xnew , Ynew])

    for i in range(Xnew ):
        for j in range(Ynew):
            value = img[i:i+size,j:j+size]
            value=value.reshape(-1) 
            value = sorted(value)
            img_new[i, j]= np.median(value )
    return  img_new  

# -----------------------------------------------Masks--------------------------------------------------------------
def Sobel(image):
    img=image
    kernelx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    kernely = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    m, n = img.shape
    img_sobelx = np.zeros([m, n])
    img_sobely = np.zeros([m, n])
    for i in range(1, m-1):
        for j in range(1, n-1):
            value = img[i-1, j-1]*kernelx[0, 0]+img[i-1, j]*kernelx[0, 1]+img[i-1, j + 1]*kernelx[0, 2]+img[i, j-1]*kernelx[1, 0]+ img[i, j]*kernelx[1, 1]+img[i, j + 1]*kernelx[1, 2]+img[i + 1, j-1]*kernelx[2, 0]+img[i + 1, j]*kernelx[2, 1]+img[i + 1, j + 1]*kernelx[2, 2]

            img_sobelx[i, j]= value
    for i in range(1, m-1):
        for j in range(1, n-1):
            value = img[i-1, j-1]*kernely[0, 0]+img[i-1, j]*kernely[0, 1]+img[i-1, j + 1]*kernely[0, 2]+img[i, j-1]*kernely[1, 0]+ img[i, j]*kernely[1, 1]+img[i, j + 1]*kernely[1, 2]+img[i + 1, j-1]*kernely[2, 0]+img[i + 1, j]*kernely[2, 1]+img[i + 1, j + 1]*kernely[2, 2]

            img_sobely[i, j]= value
    edged_img=np.sqrt( np.square(img_sobelx) + np.square(img_sobely))
    theta= np.arctan2(img_sobely,img_sobelx)
    return edged_img ,theta

def Roberts(image,kernel_size,sigma):
    img_g = Gaussian_filter(image,kernel_size,sigma)
    kernelx = np.array( [[1, 0 ],
                             [0,-1 ]] )
  
    kernely = np.array( [[ 0, 1 ],
                             [ -1, 0 ]] )
    vertical = ndimage.convolve( img_g, kernelx )
    horizontal = ndimage.convolve( img_g, kernely)
    
    edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
    return edged_img

def Prewitt(image,kernel_size,sigma):
    img_g = Gaussian_filter(image,kernel_size,sigma)
    m, n = img_g.shape
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernelx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = np.zeros([m, n])
    img_prewitty = np.zeros([m, n])
    for i in range(1, m-1):
        for j in range(1, n-1):
            value = img_g[i-1, j-1]*kernelx[0, 0]+img_g[i-1, j]*kernelx[0, 1]+img_g[i-1, j + 1]*kernelx[0, 2]+img_g[i, j-1]*kernelx[1, 0]+ img_g[i, j]*kernelx[1, 1]+img_g[i, j + 1]*kernelx[1, 2]+img_g[i + 1, j-1]*kernelx[2, 0]+img_g[i + 1, j]*kernelx[2, 1]+img_g[i + 1, j + 1]*kernelx[2, 2]

            img_prewittx[i, j]= value
    for i in range(1, m-1):
        for j in range(1, n-1):
            value = img_g[i-1, j-1]*kernely[0, 0]+img_g[i-1, j]*kernely[0, 1]+img_g[i-1, j + 1]*kernely[0, 2]+img_g[i, j-1]*kernely[1, 0]+ img_g[i, j]*kernely[1, 1]+img_g[i, j + 1]*kernely[1, 2]+img_g[i + 1, j-1]*kernely[2, 0]+img_g[i + 1, j]*kernely[2, 1]+img_g[i + 1, j + 1]*kernely[2, 2]

            img_prewitty[i, j]= value
    edged_img=np.sqrt( np.square(img_prewittx) + np.square(img_prewitty))
    return edged_img

def Canny(image,kernal_s,sigma,lowThreshold,highThreshold):
    img_g = Gaussian_filter(image,kernal_s,sigma)
    s_img, theta= Sobel(img_g)
    M, N = image.shape
    Z = np.zeros((M,N),dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255
                    
                #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = s_img[i, j+1]
                        r = s_img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = s_img[i+1, j-1]
                        r = s_img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = s_img[i+1, j]
                        r = s_img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = s_img[i-1, j-1]
                        r = s_img[i+1, j+1]

                    if (s_img[i,j] >= q) and (s_img[i,j] >= r):
                        Z[i,j] = s_img[i,j]
                    else:
                        Z[i,j] = 0

                except IndexError as e:
                    pass
   
    M, N = Z.shape
    res = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(Z >= highThreshold)
    zeros_i, zeros_j = np.where(Z < lowThreshold)

    weak_i, weak_j = np.where((Z <= highThreshold) & (Z >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    strong=255
    M, N = res.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (res[i,j] == weak):
                try:
                    if ((res[i+1, j-1] == strong) or (res[i+1, j] == strong) or (res[i+1, j+1] == strong)
                        or (res[i, j-1] == strong) or (res[i, j+1] == strong)
                        or (res[i-1, j-1] == strong) or (res[i-1, j] == strong) or (res[i-1, j+1] == strong)):
                        res[i, j] = strong
                    else:
                        res[i, j] = 0
                except IndexError as e:
                    pass
    return res










