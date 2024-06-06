import numpy as np 

def histogram_fun(image):
    histogram_array = np.bincount(image.flatten(), minlength=256)
    #normalize
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array/num_pixels
    #normalized cumulative histogram
    chistogram_array = np.cumsum(histogram_array)
    # Pixel mapping lookup table
    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
    # flatten image array into 1D list
    img_list = list(image.flatten())

    # transform pixel values to equalize
    eq_img_list = [transform_map[p] for p in img_list]

    # reshape and write back into img_array
    eq_img_array = np.reshape(np.asarray(eq_img_list), image.shape)
    
    return histogram_array,eq_img_array

def normalize_images(image):
    # initial zero ndarray
    normalized_images = np.zeros_like(image.astype(float))

    # The first images index is number of images where the other indices indicates
    # hieight, width and depth of the image
    num_images = image.shape[0]

    # Computing the minimum and maximum value of the input image to do the normalization based on them
    maximum_value, minimum_value = image.max(), image.min()

    # Normalize all the pixel values of the images to be from 0 to 1
    for img in range(num_images):
        normalized_images[img, ...] = (image[img, ...] - float(minimum_value)) / float(maximum_value - minimum_value)

    return normalized_images


def global_threshold(image):
    
    height = image.shape[0]
    width = image.shape[1]

    img_thres= np.zeros((height,width))
    thresh=np.median(image)
    # loop over the image, pixel by pixel
    for y in range(0, height):
        for x in range(0, width):
            # threshold the pixel
            pixel = image[y, x]
            img_thres[y, x] = 0 if pixel <thresh else 1
    return img_thres

def local_threshold(image):
    height = image.shape[0]
    width = image.shape[1]
    half_height = height//2
    half_width = width//2
    section1 = image[:half_height, :half_width]
    section2= image[:half_height,half_width:]
    section3= image[half_height:, :half_width]
    section4=image[half_height:,half_width:]
    img_thres= np.zeros((height,width))
    img_thres[:half_height, :half_width]=global_threshold(section1)
    img_thres[:half_height,half_width:]=global_threshold(section2)
    img_thres[half_height:, :half_width]=global_threshold(section3)
    img_thres[half_height:,half_width:]=global_threshold(section4)
    return img_thres


