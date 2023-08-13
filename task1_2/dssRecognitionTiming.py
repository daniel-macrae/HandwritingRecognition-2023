import cv2
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from data_management.augmentation.commonAug import whitespaceRemover
from skimage.transform import rotate
from tqdm  import tqdm 
import torch
import argparse
from segmentation.imageRotation import rotate_and_find_number_of_peaks, get_skew_angle
from segmentation.segmentFunction import segment_dss_page # segments BBs from an image (a whole page)
from segmentation.clustering_BBs import cluster_bounding_boxes, sort_BB_clusters_horizontally
from classification_models.DSS_Classifier import get_dss_classifier_model, classify_letters
import warnings
warnings.filterwarnings('ignore')



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
    (dim1, dim2) = img.shape
    start_time = time.time()
    img = whitespaceRemover(img=img, padding=50) # crop it down to remove all the empty space around the text
    img_width = img.shape[1]

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

    end_segment_time = time.time()
    
    

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
    num_rows = len(BB_groups_sorted)

    clustering_end_time = time.time()


    """ Classify the Letters """
    # run the classifier on each BB of each row
    # outputs a list of strings  (1 string of letters == 1 row)
    text_results = classify_letters(rotated_image, BB_groups_sorted, classifier_model, device)

    classify_end_time = time.time()


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

    segment_time = end_segment_time - start_time
    cluster_time = clustering_end_time - end_segment_time
    classification_time = classify_end_time - clustering_end_time

    
    num_characters = sum(len(i) for i in text_results)
    num_pixels = dim1 * dim2

    return segment_time, cluster_time, classification_time, num_characters, num_pixels, (dim1, dim2), num_rows, img_width

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def main(args):
    source = args.input
    output_folder = args.output
    debugging = args.debug
    create_folder_if_not_exists(output_folder)
    # make a folder to put the debugging images in
    if debugging:
        os.makedirs("debug", exist_ok = True)

    # load the classification model, for task 2, that will run on the segmented letters from task 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "classification_models/LeNet5_CNN_FINAL/model_LeNet5_bs_16-LR_5e-05_DR_0.2.pth"
    classifier_model = get_dss_classifier_model(MODEL_PATH, device)

    # if a path to a single image is passed, run once on that image
    if os.path.isfile(source): 
        print("SEGMENTING A SINGLE IMAGE")
        segment_and_classify_dss_image(source, output_folder, classifier_model, device, debugging, "debug")

    # if a path to a folder is given folder, loop through all images in the folder
    else:
        sourceFolder = source
        print("LOOPING THROUGH IMAGES FOLDER")
        segment_times = []
        clustering_times = []
        classification_times = []
        num_characters = []
        num_pixels = []
        img_dims = []
        segmentation_per_letter = []
        clustering_per_letter = []
        class_per_letter = []
        total_times = []
        numbers_of_rows = []
        image_widths = []
        #for filename in tqdm(os.listdir(sourceFolder)):
        for i in range(1):
            print(i)
            for filename in os.listdir(sourceFolder):
                input_path = os.path.join(sourceFolder, filename) # get the path to the image file
                segment_time, cluster_time, classification_time, num_character, num_pixel, img_dim, num_rows, img_width = segment_and_classify_dss_image(input_path, output_folder, classifier_model, device, debugging, "debug")
                segment_times.append(segment_time)
                clustering_times.append(cluster_time)
                classification_times.append(classification_time)
                num_characters.append(num_character)
                num_pixels.append(num_pixel)
                img_dims.append(img_dim)
                segmentation_per_letter.append(segment_time/num_character)
                clustering_per_letter.append(cluster_time/num_character)
                class_per_letter.append(classification_time/num_character)
                total_times.append(segment_time+cluster_time+classification_time)

                numbers_of_rows.append(num_rows)
                image_widths.append(img_width)
                
        print(f"SEGMENTATION - Mean time: {np.mean(segment_times):.3f}, standard deviation: {np.std(segment_times):.3f} ")
        print(f"CLUSTERING   - Mean time: {np.mean(clustering_times):.3f}, standard deviation: {np.std(clustering_times):.3f} ")
        print(f"CLASSIFIER   - Mean time: {np.mean(classification_times):.3f}, standard deviation: {np.std(classification_times):.3f} ")
        print("\n")
        print(f"SEGMENTATION   - Mean time per letter: {np.mean(segmentation_per_letter):.5f}, standard deviation: {np.std(segmentation_per_letter):.5f} ")
        print(f"CLUSTERING   - Mean time per letter: {np.mean(clustering_per_letter):.5f}, standard deviation: {np.std(clustering_per_letter):.5f} ")
        print(f"CLASSIFIER   - Mean time per letter: {np.mean(class_per_letter):.5f}, standard deviation: {np.std(class_per_letter):.5f} ")
        max_img_idx = num_pixels.index(max(num_pixels))
        min_img_idx = num_pixels.index(min(num_pixels))
        largest_img = img_dims[max_img_idx]
        smallest_img = img_dims[min_img_idx]
        print(f"Largest image = {largest_img},  Smallest = {smallest_img}")
        print(f"Most letters = {max(num_characters)},  Lowest = {min(num_characters)}")
        print("\n")
        print(f"TOTAL MEAN {np.mean(total_times):.3f} STD {np.std(total_times):.3f}")
        print(f"MEAN WIDTH {np.mean(image_widths):.3f}")
        print(f"MEAN ROWS {np.mean(numbers_of_rows):.3f}")
        print(f"MEAN N CHARACTERS {np.mean(num_characters):.3f}")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Handwriting Recognition on Dead Sea Scroll Pages')

    # Add arguments to the parser
    parser.add_argument('-i', '--input', default="test_images/", type=str, help='Input images folder. Set to \
                                                                                \'test_images\' by default.')
    parser.add_argument('-o', '--output', default="results/", type=str, help='Folder to write output. Set to \
                                                                                \'results\' by default.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    # Parse the command-line arguments
    args = parser.parse_args()
    main(args)

    




