import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from data_management.augmentation.commonAug import whitespaceRemover
from skimage.transform import rotate
from tqdm  import tqdm 

from segmentation.imageRotation import rotate_and_find_number_of_peaks, get_skew_angle
from segmentFunction import segment_dss_page # segments BBs from an image (a whole page)
from segmentation.clustering_BBs import cluster_bounding_boxes, sort_BB_clusters_horizontally
from classification_models.DSS_Classifier import get_dss_classifier_model, classify_letters

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING



def calculateLowCenters(BBs, Centers):
    LowCenters = []
    for ((x1,y1,x2,y2), old_center) in zip(BBs, Centers):
        h = y2 - y1
        w = x2 - x1

        if h/w > 1.1:
            new_center = [int((x2+x1)/2),   int(y2 - w/2)]
            LowCenters.append(new_center)
        else:
            LowCenters.append(old_center)

    return LowCenters


def write_to_document(classified_text, word_file_path):
    doc = Document()

    for row in classified_text:
        para = doc.add_paragraph(row)
        para.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    style = doc.styles['Normal']
    style.paragraph_format.line_spacing = 1

    doc.save(word_file_path)




def segment_and_classify_dss_image(sourceFolder, outputFolder, filename, classifier_model, debugging):

    """ Load the image, and trim down the white space around the text """
    img_path = os.path.join(sourceFolder, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #Read the img
    img = whitespaceRemover(img=img, padding=50) # crop it down to remove all the empty space around the text

    """ Find the optimal rotation of the page (sometimes it is skewed 2-3 degrees) """
    blurred_img, best_rotation_angle, rot_image, num_peaks = rotate_and_find_number_of_peaks(img)
    #best_rotation_angle = getSkewAngle(img)    # alternate method to find the right rotation of the page

    
    # this function rotates the original image, with the angle being defined in the counterclockwise
    rotated_image = rotate(img.copy(), best_rotation_angle, resize=True, cval=1, clip=False, mode ='constant')
    rotated_image = rotated_image * 255
    rotated_image = rotated_image.astype(np.uint8)

    if debugging:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,12))
        ax1.imshow(blurred_img)
        ax2.imshow(rotated_image)
        plt.title("Rotated " +str(-best_rotation_angle) +" degrees clockwise:  " + img_path)
        plt.show()


    """ Do the segmentation of the rotated image """

    # returns a list of bounding boxes ([x1,y1,x2,y2] each, top-left and bottom-right corner) and the center of each BB
    BBs, Centers = segment_dss_page(rotated_image)

    # in case some letters are very tall (and protrude into the line above, lower their center point)
    # if height/width > 1.1, the center point is y2 - width/2 (so the point is equidistant to the left, right and bottom of the BB, but further from the top)
    LowCenters = calculateLowCenters(BBs, Centers)
    Centers = np.array(Centers)
    LowCenters = np.array(LowCenters)

    """ Cluster the bounding boxes into rows """
    
    # the function below tries to optimise the number of clusters (using a range of values around the number of peaks detected earlier)
    # and returns the bounding boxes, sorted into rows from top to bottom (by their Y-coordinate)
    # (the clustering is done using the Y-value of each BB's 'center point')
    k_range = 3
    BB_groups = cluster_bounding_boxes(rotated_image, BBs, LowCenters, num_peaks, k_range, plot=debugging)

    # sort each row of BBs by their X-value
    BB_groups_sorted = sort_BB_clusters_horizontally(BB_groups, right_to_left=False)


    """ Classify the Letters """
    # run the classifier on each BB of each row
    # outputs a list of strings  (1 string of letters == 1 row)
    text_results = classify_letters(rotated_image, BB_groups_sorted, classifier_model)


    """ Save results to a word document """
    output_filename = filename.split('.')[0] + ".docx"
    output_file_path = os.path.join(outputFolder, output_filename)

    write_to_document(text_results, output_file_path)





def main(sourceFolder, outputFolder):

    binaryFiles = []
    for filename in os.listdir(sourceFolder):
        if 'binarized' in filename: binaryFiles.append(filename)

    TEMP_PATH = "classification_models/model_L5.pth"
    classifier_model = get_dss_classifier_model(TEMP_PATH)

    debugging = False

    for filename in tqdm(binaryFiles):
        segment_and_classify_dss_image(sourceFolder, outputFolder, filename, classifier_model, debugging)




if __name__ == "__main__":
    sourceFolder = "image-data"
    outputFolder = "Results"
    


    main(sourceFolder, outputFolder)
    




