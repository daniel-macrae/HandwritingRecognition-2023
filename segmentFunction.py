import cv2
import numpy as np



def get_circular_kernel(diameter):
    mid = (diameter - 1) / 2
    distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
    kernel = ((np.linalg.norm(distances, axis=0) - mid) <= 0).astype(np.uint8)
    return kernel


def segment_morphological_operations(image):
    image = cv2.bitwise_not(image)

    #kernel = get_circular_kernel(3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,5))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    blurred = cv2.GaussianBlur(closed, (3, 5), 0.5)

    thresholded_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #kernel = get_circular_kernel(6)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,3))
    eroded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_ERODE, kernel, iterations=1)

    return eroded_img


def find_connected_components(thresholded_img):
    im_height, im_width = thresholded_img.shape
    bounding_boxes = []
    BB_centers = []

    contours, hierarchy = cv2.findContours(thresholded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # filter out small or invalid bounding boxes
    for j, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if w >= im_width or h >= im_height:
            continue
        # these are artifacts
        if (w < 20 or h < 30):
            continue

        # sometimes parts of characters are picked out as well, control these to change what gets detected
        contour_area = cv2.contourArea(contour)
        contour_perimeter = cv2.arcLength(contour, True)
        contour_compactness = (4 * np.pi * contour_area) / (contour_perimeter ** 2)
            
        # set appropriate threshold values for area and compactness
        # min_area_threshold = 50
        max_compactness_threshold = 0.4
        #if contour_compactness > max_compactness_threshold:
        #    continue

        # 2px border for legibility
        # use max() and min() to prevent the BB going off the sides of the image
        padding = 5
        y1 = max(0, y-padding)
        y2 = min(y+h+padding, im_height)
        x1 = max(0, x-padding)
        x2 = min(x+w+padding, im_width)


        if h > 180:
            #print([x1,y1,x2,y2])
            #continue
            cutout = thresholded_img.copy()[y1:y2, x1:x2]
            midline = int((y2-y1)/2)
            cutout[midline - 2 : midline + 3, :] = 0

            [x_adjust, y_adjust] = [x1, y1]

            cutoutBBs, cutoutCenters = find_connected_components(cutout) # recursive functions are our friends :) espeically if they only recurse once

            for bb, center in zip(cutoutBBs, cutoutCenters):
                #print(center)
                [x,y] = center
                center = [x+x_adjust, y+y_adjust]
                BB_centers.append(center)

                [x1,y1,x2,y2] = bb
                bb = [x1+x_adjust, y1+y_adjust, x2+x_adjust, y2+y_adjust]
                bounding_boxes.append(bb)

        elif (y2-y1) > 0 and (x2-x1) > 0:
            BB_centers.append([int(x1+w/2),int(y1+h/2)])
            bounding_boxes.append( [x1,y1,x2,y2] )

    return bounding_boxes, BB_centers




# Takes in a grayscale image, returns the bounding boxes
def segment_dss_page(image):
    
    im_height, im_width = image.shape

    # applies morphological operations, to improve segmentation by connected components
    processed_image = segment_morphological_operations(image)

    # uses contours to find the bounding boxes, 
    # also returns the center coordinate of each bounding box
    bounding_boxes, BB_centers = find_connected_components(processed_image)

    return bounding_boxes, BB_centers
                    

    




