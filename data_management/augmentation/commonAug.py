import cv2
import random
import numpy as np
import math

def imgResizer(img, desired_size=32):
    old_size = img.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    img = cv2.resize(img, (new_size[1], new_size[0]), cv2.INTER_NEAREST)

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
    #_, new_im = cv2.threshold(new_im, 0, 255, cv2.THRESH_OTSU)

    return new_im



def whitespaceRemover(img, padding=5):  
    imHeight, imWidth = img.shape # get the image dimensions
    # find the BB of the contents of the image
    gray_img = img.copy()
    gray_img = 255*(gray_img < 128).astype(np.uint8) # To invert the text to white
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8)) # Perform noise filtering
    coords = cv2.findNonZero(gray_img) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box

    # pad each side
    y_min = y-padding if y-padding>=0 else 0
    y_max = y+h+padding if y+h+padding<=imHeight else imHeight
    x_min = x-padding if x-padding>=0 else 0
    x_max = x+w+padding if x+w+padding<=imWidth else imWidth

    cropped_img = img[y_min:y_max, x_min:x_max] # Crop the image - note we do this on the original image
    
    return cropped_img


# warp the letter images, parameters are slightly different to the whole page image warper (as these letter images are smaller)
def letterImageWarper(img):
    rows, cols = img.shape

    warp_factors = list(np.arange(-5,6,1))
    wave_lengths = list(np.arange(1,2.2,0.1))

    horizontalWarpFactor = random.choice(warp_factors)
    verticalWarpFactor = random.choice(warp_factors)
    horzWaveLength = random.choice(wave_lengths)
    vertWaveLength = random.choice(wave_lengths)

    img_output = np.ones(img.shape, dtype=img.dtype) * 255 # make a blank white image to place the warp on

    x_offsets = [int(horizontalWarpFactor * math.sin(2 * 3.14 * x_coord / (cols*horzWaveLength))) for x_coord in np.arange(0,cols,1)] 
    y_offsets = [int(verticalWarpFactor * math.sin(2 * 3.14 * y_coord / (rows*vertWaveLength))) for y_coord in np.arange(0,rows,1)] 


    for i in range(rows):
        for j in range(cols):

            offset_x = x_offsets[i] # how much to move left and right (which is based on row positon)
            offset_y = y_offsets[j] # how much to move up and down (which is based on column positon)

            # if still within the image, move the pixel value, otherwise, the output image is white anyway (255 value)
            if 0 <= i+offset_y < rows  and   0 <= j+offset_x < cols:
                img_output[i,j] = img[i+offset_y, j+offset_x]

    return img_output


# function to rotate an image slighty (randomly, within a range)
def imageRotator(img, rotation_range=15):
    angle = random.randint(5, rotation_range) # at least 5 degrees of rotation, prevents images that are rotated 0 degrees
    if random.random() > 0.5:
        angle *= -1  # 50% chance of rotating the other direction


    rows, cols = img.shape

    img_center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(img_center, angle, 1)
    rotated_image = cv2.warpAffine(img, M, (cols, rows),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255))
    
    return rotated_image



# function to shear an image slighty in the x and y direction (randomly, within a range)
def imageShearer(img, shear_range=0.2):
    Xshear = random.uniform(-shear_range, shear_range)
    Yshear = random.uniform(-shear_range, shear_range)

    rows, cols = img.shape

    M = np.float32([[1, Xshear, 0],
             	[Yshear, 1  , 0],
            	[0, 0  , 1]])
    
    sheared_image = cv2.warpPerspective(img, M, (cols, rows),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255))
    
    return sheared_image


def imageEroder(img, max_erode_size = 3):
    size = random.randint(2, max_erode_size)
    kernel = np.ones((size, size), np.uint8)
  
    # Using cv2.erode() method 
    img = cv2.erode(img, kernel) 

    return img


def imageDilator(img, max_dilate_size = 3):
    size = random.randint(2, max_dilate_size)
    kernel = np.ones((size, size), np.uint8)
  
    # Using cv2.erode() method 
    img = cv2.dilate(img, kernel) 

    return img