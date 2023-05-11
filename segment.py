import cv2
import os
from pprint import pprint as print
import numpy as np


class Segment:
    def __init__(self, input_folder=os.path.join(os.getcwd(),"image-data/image-data")):
        self.input_folder = input_folder
        self.output_folder = os.path.join(os.getcwd(),'Data/Segmented')
        os.makedirs(self.output_folder, exist_ok=True)

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
                # cv2.imshow('OGimage',img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                print(f"processing image {i} out of {len(images)}")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
                # open
                morph = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
                # close
                morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
                # dilate 
                morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel)
                ret, labels = cv2.connectedComponents(morph)
                contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for label in range(1, ret):
                    component = np.uint8(labels == label) * 255
                    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    # Filter out small or invalid bounding boxes
                    for j,contour in enumerate(contours):
                        (x, y, w, h) = cv2.boundingRect(contour)
                        # these are artifacts
                        if w < 10 or h < 10:
                            continue
                        # 2px border lor legibility
                        roi = img[y-2:y+h+2, x-2:x+w+2]
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        filename = self.output_folder + f'/page_{i}_character_{label}{j}.png'
                        # cv2.imwrite(filename,roi)
                
                imS = cv2.resize(img, (1000, 800)) 
                cv2.imshow('Segmented Characters', imS)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                # for j,contour in enumerate(contours):
                #     # get bounding box
                #     (x, y, w, h) = cv2.boundingRect(contour)
                #     # 2px border for clarity
                #     roi = img[y-2:y+h+2, x-2:x+w+2]
                #     # smaller than this and its not a whole letter, just noise
                #     if len(roi)>10:
                #         filename = self.output_folder + f'/page_{i}_character_{j}.png'
                #         cv2.imwrite(filename,roi)
                    
        else:
            print('Could not load dataset')

    def loadGeneratedData(self):
        try:
            files = os.listdir(os.path.join(self.input_folder,"Data/Habbakuk_Pages"))
        except FileNotFoundError:
            print("Incorrect path. Please provide full path to input folder")
            return []   
        
        files = [os.path.join(self.input_folder, file) for file in files]
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

        files = [os.path.join(self.input_folder, file) for file in files]
        images = []
        for file in files:
            image = cv2.imread(file)
            images.append(image)
        return images

    


if __name__ == "__main__":
    a = Segment()
    #a = CharacterSegment("/home/lucypher/Desktop/HandRec/Project/image-data/image-data")
    a.segment_characters()




