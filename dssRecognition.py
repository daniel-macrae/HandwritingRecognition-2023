import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from data_management.augmentation.commonAug import whitespaceRemover
from skimage.transform import rotate
from tqdm  import tqdm 
import torch

from segmentation.imageRotation import rotate_and_find_number_of_peaks, get_skew_angle
from segmentFunction import segment_dss_page # segments BBs from an image (a whole page)
from segmentation.clustering_BBs import cluster_bounding_boxes, sort_BB_clusters_horizontally
from classification_models.DSS_Classifier import get_dss_classifier_model, classify_letters

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING


def plotSegmentedBBs(img, BBs):
    for (x1,y1,x2,y2) in BBs:
        img = cv2.rectangle(img, (x1,y1), (x2,y2), color = (0,0,0), thickness=2)
    return img

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







"""
Function for the full Task 1 and 2 pipeline
"""


def segment_and_classify_dss_image(input_path, outputFolder, classifier_model, device, debugging, debugging_folder):

    """ Load the image, and trim down the white space around the text """
    #img_path = os.path.join(sourceFolder, filename)
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE) #Read the img
    img = whitespaceRemover(img=img, padding=50) # crop it down to remove all the empty space around the text

    """ Find the optimal rotation of the page (sometimes it is skewed 2-3 degrees) """
    blurred_img, best_rotation_angle, rot_image, num_peaks = rotate_and_find_number_of_peaks(img)
    #best_rotation_angle2 =     # alternate method to find the right rotation of the page
    best_rotation_angle += -get_skew_angle(img)
    best_rotation_angle = int(best_rotation_angle/2)
    print(best_rotation_angle)
    
    # this function rotates the original image, with the angle being defined in the counterclockwise
    rotated_image = rotate(img.copy(), best_rotation_angle, resize=True, cval=1, clip=False, mode ='constant')
    rotated_image = rotated_image * 255
    rotated_image = rotated_image.astype(np.uint8)


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
    BB_groups, clustering_output_image = cluster_bounding_boxes(rotated_image, BBs, LowCenters, num_peaks, k_range, plot=debugging)

    # sort each row of BBs by their X-value
    BB_groups_sorted = sort_BB_clusters_horizontally(BB_groups, right_to_left=False)


    """ Classify the Letters """
    # run the classifier on each BB of each row
    # outputs a list of strings  (1 string of letters == 1 row)
    text_results = classify_letters(rotated_image, BB_groups_sorted, classifier_model, device)

    """ Save results to a word document """
    filename = os.path.split(input_path)[-1].split('.')[0]
    output_filename = filename + ".docx"
    output_file_path = os.path.join(outputFolder, output_filename)

    write_to_document(text_results, output_file_path) 


    """ IF DEBUGGING, save seperate images of the rotating of the image, and segmentation and clustering of the BBs"""
    # saves intermediary results to a debugging folder
    if debugging:
        # save the rotation result
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,12))
        ax1.imshow(blurred_img, cmap='gray')
        ax1.set_title("blurred image")
        ax2.imshow(img, cmap='gray')
        ax2.set_title("original image")
        ax3.imshow(rotated_image, cmap='gray')
        ax3.set_title("rotated image")
        plt.title("Rotated " +str(-best_rotation_angle) +" degrees clockwise")
        
        path = os.path.join(debugging_folder, filename + '_ROT.jpg')
        plt.savefig(path)
        plt.clf()


        # save the sementation image
        segmentation_image = plotSegmentedBBs(rotated_image.copy(), BBs)
        path = os.path.join(debugging_folder, filename + '_SEG.jpg')
        cv2.imwrite(path, segmentation_image)

        # save the clustering image (coloured BBs w. the horizontal lines of each cluster)
        path = os.path.join(debugging_folder, filename + '_CLUST.jpg')
        cv2.imwrite(path, clustering_output_image)

        # save a txt file of the results
        # saving a txt file doesn't seem to work with the hebrew letters, so just print them for now
        #print(text_results)
        path = os.path.join(debugging_folder, filename + '_RESULTS.docx')
        write_to_document(text_results, path) 
        #with open(path, 'w') as f:
        #    for line in text_results:
        #        f.write(f"{0}\n")





def main(args):
    source = args.input
    output_folder = args.output_folder

    debugging = args.debugging
    debugging_folder = args.debugging_folder

    print(source, output_folder)
    print(debugging, debugging_folder, '\n')

    # make a folder to put the debugging images in
    if debugging:
        os.makedirs(debugging_folder, exist_ok = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEMP_PATH = "classification_models/LeNet5_FINALmodel/model_LeNet5_bs_16-LR_5e-05_DR_0.2.pth"
    classifier_model = get_dss_classifier_model(TEMP_PATH, device)

    # if image, run once
    if os.path.isfile(source): 
        print("SEGMENTING A SINGLE IMAGE")
        segment_and_classify_dss_image(source, output_folder, classifier_model, device, debugging, debugging_folder)

    # if folder, loop through all images in the folder
    else:
        sourceFolder = source
        print("LOOPING THROUGH FOLDER")
        for filename in tqdm(os.listdir(sourceFolder)):
            #print(filename)

            input_path = os.path.join(sourceFolder, filename) # get the path to the image file
            segment_and_classify_dss_image(input_path, output_folder, classifier_model, device, debugging, debugging_folder)




def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
    parser.add_argument("--input", default="Data/image-data", type=str, help="path to input image or folder of images")
    parser.add_argument("--output_folder", default="Results", type=str, help="folder to save the results in")
    
    parser.add_argument("--debugging", default=True, type=bool, help="whether to save images of the intermediate steps")
    parser.add_argument("--debugging_folder", default="debug", type=str, help="folder to save the debugging images in")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()    

    main(args)
    




