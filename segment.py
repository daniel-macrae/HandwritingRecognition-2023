import cv2
import os
from pprint import pprint as print
import numpy as np
from data_management.augmentation.commonAug import whitespaceRemover




class Segment:
    def __init__(self, input_folder=os.path.join(os.getcwd(),"image-data")):
        self.input_folder = input_folder
        self.output_folder = os.path.join(os.getcwd(),'Data/Segmented')
        os.makedirs(self.output_folder, exist_ok=True)

    def segment_page(self, image=None, path=''):
        image_name = f"page_"
        if "image-data" in path:
            image_name = path.split('/')[-1][:-4]
            image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = whitespaceRemover(img=gray, padding=50)


        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        # open
        morph = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        # close
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
        
        # dilate 
        morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel)
        ret, labels = cv2.connectedComponents(morph)
        contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for label in range(1, ret):
            component = np.uint8(labels == label) * 255
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # filter out small or invalid bounding boxes
            for j,contour in enumerate(contours):
                    (x, y, w, h) = cv2.boundingRect(contour)
                    # these are artifacts
                    if (w < 30 or h < 30):
                        continue

                    # sometimes parts of characters are picked out as well, control these to change what gets detected
                    contour_area = cv2.contourArea(contour)
                    contour_perimeter = cv2.arcLength(contour, True)
                    contour_compactness = (4 * np.pi * contour_area) / (contour_perimeter ** 2)
                        
                    # set appropriate threshold values for area and compactness
                    # min_area_threshold = 50
                    max_compactness_threshold = 0.4
                    if contour_compactness > max_compactness_threshold:
                        continue
                    # 2px border for legibility
                    roi = image[y-2:y+h+2, x-2:x+w+2]
                    filename = self.output_folder + f'/{image_name}_character_{j}_x={x}_y={y}_w={w}_h={h}.png'
                    if np.all(np.array(roi.shape) > 0):
                        cv2.imwrite(filename,roi)


    def segment_characters(self, real_data=True, generated_data=False):
        if real_data:
            images = self.loadRealData()
        if generated_data:
            images = self.loadGeneratedData()
        if real_data and generated_data:
            print("Choose one dataset to segment at one time")
            pass
        if images:
            for i,img in enumerate(images):

                print(f"processing image {i+1} out of {len(images)}")
                if np.any(np.array(img.shape) < 0):
                    continue
                self.segment_page(img)
    
        else:
            print('Could not load dataset')

    def loadGeneratedData(self):
        try:
            files = os.listdir("Data/Habbakuk_Pages/images")
        except FileNotFoundError:
            print("Incorrect path. Please provide full path to input folder")
            return []
        files = [os.path.join("Data/Habbakuk_Pages/images",file) for file in files]   
        images = []
        for file in files:
            image = cv2.imread(file)
            images.append(image)
        return images

    def loadRealData(self, only_binary=True, only_fused=False):
        try:
            files = os.listdir(self.input_folder)
        except FileNotFoundError:
            print("Incorrect path. Please provide full path to input folder")
            return []       
        if only_binary:
            files = [file for file in files if "binarized" in file]
        if only_fused:
            files = [file for file in files if "fused" in file]
        if only_binary and only_fused:
            pass

        files = [os.path.join(self.input_folder, file) for file in files]
        images = []
        for file in files:
            image = cv2.imread(file)
            images.append(image)
        return images

    




