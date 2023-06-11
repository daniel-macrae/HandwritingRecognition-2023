import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from data_management.augmentation.commonAug import whitespaceRemover
from skimage.transform import rotate
from tqdm  import tqdm 
import torch

from segmentation.imageRotation import rotate_and_find_number_of_peaks, get_skew_angle
from segmentation.segmentFunction import segment_dss_page # segments BBs from an image (a whole page)
from segmentation.clustering_BBs import cluster_bounding_boxes, sort_BB_clusters_horizontally
from classification_models.DSS_Classifier import get_dss_classifier_model, classify_letters

# function that helps visualise the segmentation, for the 'debugging' option
def plotSegmentedBBs(img, BBs):
    for (x1,y1,x2,y2) in BBs:
        img = cv2.rectangle(img, (x1,y1), (x2,y2), color = (0,0,0), thickness=2)
    return img

# better center points for tall letters (for clustering)
def calculateLowCenters(BBs, Centers):
    LowCenters = []
    for ((x1,y1,x2,y2), old_center) in zip(BBs, Centers):
        h = y2 - y1
        w = x2 - x1

        if h/w > 1.1: # if the character is tall, set the 'center' to be lower
            new_center = [int((x2+x1)/2),   int(y2 - w/2)]
            LowCenters.append(new_center)
        else: # use the exact center coordinate of the bounding box
            LowCenters.append(old_center)

    return LowCenters

# function to write the result files
def write_to_txt_file(classified_text, txt_file_path):
    with open(txt_file_path, 'wb') as f:
        for line in classified_text:
            line = line +'\n'
            encoded_result = line.encode("UTF-8")
            f.write(encoded_result)



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
    best_rotation_angle += -get_skew_angle(img) # alternate method to find the right rotation of the page
    best_rotation_angle = int(best_rotation_angle/2) # averaging them usually gives good results
    
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


    """  Save the results to a txt file  """
    filename = os.path.split(input_path)[-1].split('.')[0]
    output_filename = filename + "_characters.txt"
    output_file_path = os.path.join(outputFolder, output_filename)
    write_to_txt_file(text_results, output_file_path)
    
    
    



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
        path = os.path.join(debugging_folder, filename + '_RESULTS.txt')
        write_to_txt_file(text_results, path) 




def main(source, output_folder, debugging, debugging_folder):

    os.makedirs(output_folder, exist_ok = True)

    # make a folder to put the debugging images in
    if debugging:
        os.makedirs(debugging_folder, exist_ok = True)

    # load the classification model, for task 2, that will run on the segmented letters from task 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "classification_models/LeNet5_CNN_FINAL/model_LeNet5_bs_16-LR_5e-05_DR_0.2.pth"
    classifier_model = get_dss_classifier_model(MODEL_PATH, device)

    # if a path to a single image is passed, run once on that image
    if os.path.isfile(source): 
        print("SEGMENTING A SINGLE IMAGE")
        segment_and_classify_dss_image(source, output_folder, classifier_model, device, debugging, debugging_folder)

    # if a path to a folder is given folder, loop through all images in the folder
    else:
        sourceFolder = source
        print("LOOPING THROUGH IMAGES FOLDER")
        for filename in tqdm(os.listdir(sourceFolder)):
            input_path = os.path.join(sourceFolder, filename) # get the path to the image file
            segment_and_classify_dss_image(input_path, output_folder, classifier_model, device, debugging, debugging_folder)




if __name__ == "__main__":
    
    input_folder = sys.argv[1] # get the input folder from the bash command (first argument)

    # if another argument is passed, then save the debugging images
    try: 
        x = (sys.argv[2])
        debugging = True
    except:
        debugging = False

    output_folder = './results'
    debugging_folder = "./debug"

    print("\nInput folder:  ", input_folder)
    print("Output folder: ", output_folder)
    print("Saving intermediate results in '"+ str(debugging_folder)+"' folder:", debugging,'\n')
    

    main(input_folder, output_folder, debugging, debugging_folder)

    print("\nDONE!")
    print("Results, as txt files, can be found in the './results' folder")
    




