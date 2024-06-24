# CV-A01: Images, Filters, Histograms, Gradients, Frequency

## Description:
A small web application based app developed with python and streamlit, to apply different image processing techniques.

## Requirements:
• Python 3.  
• Streamlit 1.13.0  
• Numpy 1.23.4  
• Scipy 1.9.2  
• Matplotlib 3.6.2  
• Seaborn 3.6.2  

## Running command:
Streamlit run server.py 

-The UI contains three main tabs filtering, histogram and hybrid images.

# Tab1:
• Noise  
• Filters  
• Edge masks
### Noise:
1. Uniform noise
Using uniform distribution  
![Screenshot (1437)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/c57280a0-206b-48a5-af6a-e917b9ca9688)  
Result  
![Screenshot (1438)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/df5e17a8-34f1-41d3-a9cc-bce98a73be66)

2. Gaussian noise  
Using Gaussian distribution
![Screenshot (1439)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/c22dab35-d545-47a4-9c70-531ebd14f940)  
Result  
![Screenshot (1440)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/0b0dbb43-a038-4fb7-8ef7-110ea20978b5)

3. Salt and pepper noise  
Result  
![Screenshot (1441)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/cd7d1509-c03a-408a-9714-af7bbaa7eb65)

### Filters:  
1. Average filter  
It uses 3 x 3 kernel which is convolved with the input image to calculate output.
![Screenshot (1442)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/27f79e6e-720d-427c-8be7-ca0bbc82bd2d)  
Parameters: Kernel_size  
Result  
![Screenshot (1443)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/ca425ae1-1fa0-4865-815d-ddc0fd9b49ed)  

2. Gaussion filter  
The formula to design gaussian kernel.   
![Screenshot (1444)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/3865e52b-7a6e-46dc-9641-860e301c82b4)   
Example:   
![Screenshot (1445)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/3e3c2394-13f5-4708-8ddb-152ab5b604b7)  
It uses gaussian kernel which is convolved with the input image to calculate output.  
Parameters: Kernel_size,sigma  
Result  
![Screenshot (1446)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/240a86bc-f956-44b1-ac71-fe70da99cbc1)  

3. Median filter  
![Screenshot (1447)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/54091c6c-4d9f-4fd2-ba61-3cdc266dddf5)  
Parameters: Kernel_size  
Result   
![Screenshot (1448)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/d856bc89-c41e-488e-90fe-d88333f58b0d)  

Note    
-The larger the kernel size, the greater the smoothing.      
-Median filter is the best filter for salt and pepper noise.    

### Edges:
Edges are significant changes in intensity of pixels of digital image. There are 
3types of edges: horizontal, vertical and diagonal. Edge detection is segmentation 
into regions of discontinuity, there are 2 types of operators:  
 Gradient: compute 1st order derivative like Sobel, Prewitt and Roberts.  
 Gaussian: compute 2nd order derivative like Canny.     
1. Sobel:    
discrete differentiation operator. It computes the gradient approximation of image intensity. It 
uses two 3 x 3 kernels or masks which are convolved with the input image to calculate the vertical 
and horizontal derivative approximations.    
![Screenshot (1449)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/ce81cccc-6511-4e8e-b433-5385040dbadc)  
Result  
![Screenshot (1450)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/836867cb-76a1-496a-a2df-4228f479a2a6)  
2. Prewitt:  
This operator is almost similar to the Sobel operator. It also detects verticaland horizontal edges of an image. It is one of the best ways to detect theorientation and magnitude of an image.  
![Screenshot (1451)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/c6a35b0e-de51-404a-a663-5485f1543cb0)  
Result  
![Screenshot (1452)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/ec4f3917-0dcc-4057-b249-908d536598ea)
3. Roberts:   
This operator computes the sum of squares of the differences between diagonally adjacent pixels in an image through discrete differentiation. Then approximation is made by 
convolving the image with 2 x 2 kernels.  
![Screenshot (1453)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/2b094248-4d6b-42f7-8e49-c51a5ee4a23b)  
Result  
![Screenshot (1454)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/1a894eec-f01a-4838-a425-5e2486107af0)  
4. Canny:  
This operator is not susceptible to noise. It extracts image features without 
affecting the feature. Canny edge detector have advanced algorithm on 5 
steps:  
• Noise reduction by Gaussian filter.   
• Gradient calculation by sobel mask.  
• Non-maximum suppression: thin out the edges.  
• Double threshold: identifying 3 kinds of pixels: strong, weak, and 
non-relevant  
• Edge Tracking by Hysteresis:  
transforming weak pixels into strong ones, if and only if at least 
one of the pixels around the one being processed is a strong one.  
 By changing the input lower and higher thresholds the edges changes.

Result  
![Screenshot (1455)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/e53bcc80-94da-40bf-96b7-c73b1f56a346)  

# Tab2:
• Histogram, its distribution curve, Equalization and Normalization  
• Local and Global Thresholding  
• RGB Visualization  
### 1. Histogram and its distribution curve:
An image histogram is a representation of the range of tonal values that is present in an image.    
![Screenshot (1456)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/ae733157-d20f-4cb0-acd3-f0227be5e114)  

### 2. Equalization:
Histogram equalization is a method in image processing of contrast adjustment 
using the image’s histogram.  
#### Algorithm   
we calculate the normalized histogram of the image. Normalization is performed 
by dividing the frequency of each bin by the total number of pixels in the image
then we derive a lookup table which maps the pixel intensities to achieve an 
equalized histogram characteristics. Finally Transform pixel intensity of the 
original image with the lookup table.  
![Screenshot (1457)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/20f46c57-d9b2-4def-8b7b-90640e1ad755)  

### 3. Normalization:
The process that changes the range of pixel intensity values to be from 0 to 1  
![Screenshot (1458)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/6c22c590-47a3-4470-b1d3-55f2652262a5)  

#### Algorithm  
we normalize a color input image using min-max norm. The image pixel values are
normalized to a range [0,1]  

### 4. Local and Global Thresholding:
In global thresholding, each pixel value in the image is compared with a single 
(global) threshold value( higher than the threshold the pixel value will be 1 
otherwise it will be 0). In local thresholding the same happens however the image 
is divided into parts each part has its own threshold based on the pixel values in 
that part.    
![Screenshot (1459)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/64b1c838-9064-4369-a3c5-7440dfedbbe0)  

#### Algorithm
In global thresholding algorithm we create an empty array as the same size as the 
image and we calculate the median of an image as our threshold we loop over 
pixels values to decide each pixel value (either 1 or 0). In local thresholding we do 
the same to each part of the image.  
### o RGB Channels:
We read image in color mood and split it to 3 channel ,then plot them.
o RGB Channels & Cumulative curves for input image.  
![Screenshot (1460)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/0430652b-6b91-4cea-9b6c-531a9f3d63bc)  
o Cumulative Curve for Equalized image  
![Screenshot (1461)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/176dbb5e-9fe2-4a54-a7d7-34270c6cda62) 

# Tab3:
• Frequency Domain Filter  
• Hybrid Images  

### Frequency Domain Filter:
• First transform the image F(u,v) = F{f(u,v0} into frequency
domain using fourier’s transform.  
• Multiply the filter H(u,v).  
• Take the inverse fourier’s transform g(x,y) = f-1
{g(u,v)}.  
#### 1. Ideal Low Pass Filter
Low frequency components are located at corners, we shift them first to the center then apply the filter. The filter 
eliminates all high frequencies and the resulted image is smoothed.  
![Screenshot (1462)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/794eb1d7-24bd-40fb-9e13-0a8c859758ce)  
Result     
![Screenshot (1463)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/5d9442fe-888a-41c2-afeb-00f36545fe6b)  
#### 2. Ideal high pass filter
It does the opposite of LPF so simply we can create it 
by subtracting the LPF from 1 (HP = 1- LP). 
The filter removes the constant brightness regions 
and leave the regions with rapid brightness 
transitions.  
![Screenshot (1464)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/4a5c9846-6f0b-4bf6-8841-ff223a156b4c)  
Result  
![Screenshot (1465)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/228b1168-499e-4bb4-a339-a63f02a69281)  
##### parameters
 The parameter D0 represents the radius of the cut off circle as it get larger the image will 
be less smoothed in LPF and more sharper in HPF.  
###  Hybrid Image:
We upload 2 images apply high & low filter to them,
then choose option :  
• Low pass image1 + High pass image2  
• High pass image1 + Low pass image2  
o Uploaded Images  
![Screenshot (1466)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/59ffd716-08bf-4195-9231-de0e21d0d1e4)  

##### 1) High pass image1 + Low pass image2
![Screenshot (1467)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/e86cc745-c664-4163-b106-16fc0bf5f513)  
##### 2) Low pass image1 + High pass image2
![Screenshot (1469)](https://github.com/MayarFayez/CV_-HPF-LPF_-Techniques-/assets/93496610/f1861000-5401-42c7-a267-f3303197dff8)












































