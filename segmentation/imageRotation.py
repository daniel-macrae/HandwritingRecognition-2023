
from segmentation.peakdetect import *
from skimage.transform import rotate
import os
from skimage.io import imread, imsave
import numpy as np
import cv2
import matplotlib.pyplot as plt
from data_management.augmentation.commonAug import imgResizer


def write_image(image, filename, runmode=1):
    path_output = os.path.abspath(filename)
    #image = image.astype(np.uint8)
    cv2.imwrite(filename=path_output,img=image)




def rotate_and_find_number_of_peaks(img):
    blurredImg = img.copy()
    # apply mutliple iterations of dilation and erosion with a rectangular kernel
    # to try and remove vertical lines, and thicken horizontal lines
    for i in range(8):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
        blurredImg = cv2.dilate(blurredImg, kernel, iterations=1)

        blurredImg = cv2.erode(blurredImg, kernel, iterations=1)
    
    # make the image smaller (makes finding the rotations much faster)
    blurredImg = imgResizer(blurredImg, desired_size = 500)

    # soften the edges
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #blurryboi = cv2.erode(blurryboi, kernel, iterations=1)
    blurredImg = cv2.GaussianBlur(blurredImg, (9, 9), 0)
    
    rot_image, rot_degree, num_peaks, rot_line_peaks = find_optimal_rotation(blurredImg, "hi", lookahead=15, runmode=0)

    return blurredImg, rot_degree, rot_image, num_peaks





def find_optimal_rotation(image, output_directory, lookahead=20, min_degree=-10, max_degree=10, runmode=1):
    optimum_rot_degree = -90
    optimum_score = 0
    optimum_rot_image = image
    optimum_rot_line_peaks = []
    optimum_num_peaks = 1

    for degree in range (min_degree, max_degree):

        rotated_image = rotate(image, degree, resize=True, cval=1, clip=False, mode ='constant')
        
        # Rotate results in a normalised floating image -> convert it to uint8
        #rotated_image = rotated_image * 255
        #rotated_image = rotated_image.astype(np.uint8)
        
        if runmode > 0: # Show intermediary rotated images in normal and debug mode
            write_image(rotated_image, output_directory + '/rotated_' + str(degree) + '.jpg', runmode=runmode)

        # 1 = column reduction.
        # CV_REDUCE_AVG instead of sum, because we want the normalized number of pixels
        histogram = cv2.reduce(rotated_image, 1, cv2.REDUCE_AVG)
        # Transpose column vector into row vector
        histogram = histogram.reshape(-1)

        if (runmode > 0): # Show intermediary histograms in normal and debug mode
            plt.plot(histogram)
            plt.title('Degree=' + str(degree))
            plt.savefig(output_directory+'/histogram_' + str(degree) + '.jpg')
            plt.clf()

        line_peaks = peakdetect(histogram, lookahead=lookahead)

        # sometimes the number of positive peaks is not identical to the number of negative peaks
        number_peaks = min(len(line_peaks[0]), len(line_peaks[1]))
        
        score = 0
        if number_peaks != 0:
            for peak in range(0, number_peaks):
                score += line_peaks[0][peak][1] - line_peaks[1][peak][1]
            
            score = score / number_peaks
        #except:
        #    score = 0

        if runmode > 1: # Show tested degrees in debug mode only
            print ('Degree=' + str(degree) + '; Score=' + str(score))

        if score >= optimum_score and abs(degree) <= abs(optimum_rot_degree):
            optimum_score = score
            optimum_rot_degree = degree
            optimum_rot_image = rotated_image
            optimum_rot_line_peaks = line_peaks
            optimum_num_peaks = number_peaks

    return optimum_rot_image, optimum_rot_degree, optimum_num_peaks, optimum_rot_line_peaks






"""
METHOD 2
"""



def get_skew_angle(gray):
    # Prep image, copy, convert to gray scale, blur, and threshold
    blur = cv2.GaussianBlur(gray, (25, 25), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilation to merge letters into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle

    # in case the angle is massive, its because the image is much taller than it is wide
    # this -=90 is to stop it from trying to turn the image sideways
    if angle > 15: 
        angle -= 90
        #return 0
    return int(-1.0 * angle)



