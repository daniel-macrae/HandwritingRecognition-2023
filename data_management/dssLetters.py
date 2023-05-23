import os
from collections import Counter
import shutil
from sklearn.model_selection import train_test_split
from data_management.augmentation.habbakukGenerator import create_letter_image
import cv2
import random
from itertools import chain, combinations
from data_management.augmentation.commonAug import imgResizer, imageRotator, imageShearer, letterImageWarper, imageDilator, imageEroder
from tqdm import tqdm


# function that copies images to a temporary folder, only if the image is unique enough
# similarity between images measured as the difference in position and image size
# if the difference is below the threshold in both x-position, y-position, width, and height, then do not use this image
def duplicates_remover(source_folder, temp_folder, positionThreshold, sizeThreshold):
    shapesDict = {}
    
    for folder in tqdm(os.listdir(source_folder)):
        folder_path = os.path.join(source_folder, folder) # current folder directory
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename) # full path to the image we are currently on
            if os.path.isdir(file_path):
              continue
            target_folder = os.path.join(temp_folder, folder) # stuff to make a target directory
            os.makedirs(target_folder, exist_ok = True)
            target_file_path = os.path.join(target_folder, filename)

            split = filename.split("-")[9:13]  # get info about this image
            try: # there are two files in the whole monkbrill2 folder that have a different naming convention (but they are both safe to keep without dataset leakage)

                x_pos, y_pos = int(split[0].split("=")[1]), int(split[1].split("=")[1])
                width, height = int(split[2].split("=")[1]), int(split[3].split("=")[1])
            
                extract_file = filename.split("-")[2]  # which DSS extract this letter image came from

                if extract_file in shapesDict.keys():  # if we're on an image that came from the same DSS extract
                    seen_before = False
                    for know_img in shapesDict[extract_file]:
                        [known_x, known_y, known_width, known_height] = know_img
                        # if their x, y positions are similar, AND they have similar image dimensions; ignore this image
                        if abs(known_x - x_pos) < positionThreshold and abs(known_width - width) < sizeThreshold:
                            if abs(known_y - y_pos) < positionThreshold and abs(known_height - height) < sizeThreshold:
                                seen_before = True
                                break
                
                    if not seen_before:
                        shapesDict[extract_file].append([x_pos, y_pos, width, height])
                        shutil.copy(file_path, target_file_path)
                    else: 
                        pass # image is ignored, its too similar to one we have already

                else:
                    shapesDict[extract_file] = [[x_pos, y_pos, width, height]]                
                    shutil.copy(file_path, target_file_path)
            except:
                shutil.copy(file_path, target_file_path)





"""
make train, test set. validation set is optional
"""

def train_test_val_splitter(temp_folder, outputFolder, validation_set = True, reshape = True, desired_size = 50):
    # put the paths to all of the images here
    filePaths = []
    for root, dirs, files in os.walk(temp_folder):
        for name in files:
            filePaths.append(os.path.join(root, name))


    # get the class label for each image in filePaths
    image_labels = [file_path.split('/')[-2].split("_")[-1] for file_path in filePaths]
                        #changed '\\' for '/'
                        
    # make train, validate, testing sets (using the image paths and labels)
    X_train, X_test, y_train, y_test = train_test_split(filePaths, image_labels, test_size=0.3, random_state=0, stratify=image_labels, shuffle=True)

    if validation_set:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train, shuffle=True)

    
    sets = {"test" : (X_test, y_test),
            "train" : (X_train, y_train) 
            }
    
    if validation_set:  sets["validation"] = (X_val, y_val)

    # make train/test(/validation) folders
    for key in sets:
        folderPath = os.path.join(outputFolder, key) # makes the train, validate, test folders
        os.makedirs(folderPath, exist_ok = True)


    # move the files into the folders
    # also resize to 50x50 here, may as well do it now
    for key in sets:
        print(key)
        (filePaths, image_labels) = sets[key]
        print("   ", Counter(image_labels))

        for imgPath, label in zip(filePaths, image_labels):
        
            folderPath = os.path.join(os.path.join(outputFolder, key), label)
            os.makedirs(folderPath, exist_ok = True)  # checks to see if the folder for this class is made (within the train/val or test folders)

            filename = os.path.split(imgPath)[-1]
            
            if not filename.lower().endswith('.jpg'):
                print(f"Skipping file {imgPath} - not a JPEG image")
                continue
                
            target_im_dir = os.path.join(folderPath, filename)
            #print(target_im_dir)

            if reshape:
                img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
                img = imgResizer(img, desired_size=desired_size)  # reshapes to 50x50
                cv2.imwrite(target_im_dir, img)
            else:
                shutil.copy(imgPath, target_im_dir)

    # delete the temporary folder, we're done with it here
    shutil.rmtree(temp_folder)





""""
DEALING WITH CLASS IMBALANCES
"""

def instancesPerClassCounter(folder):
    classCounts = {}
    for content in os.listdir(folder):
        folderPath = os.path.join(folder, content)
        if os.path.isdir(folderPath):
            classCounts[content] = len(os.listdir(folderPath))

    return classCounts





training_set_folder = "Data/dssLetters/train/"



def fillInImbalancedClasses(training_set_folder, amount=1):

    letterClassCounts = instancesPerClassCounter(training_set_folder)

    print(letterClassCounts)

    highestCount = max(letterClassCounts.values())

    print("letters added per class:")
    for letterClass in os.listdir(training_set_folder):
        n = 0
        folderPath = os.path.join(training_set_folder, letterClass)
        classes = list(letterClassCounts.keys())
        #print(letterClassCounts.keys())

        while letterClassCounts[letterClass] < highestCount * amount:
            # generate a new image, using the habbakuk font
            img = create_letter_image(letterClass, (50,50))
            img = letterImageWarper(img)
            ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

            labelIDX = classes.index(letterClass)

            # make the filename, save the image (the label and label index is contained in the filename, so no need for seperate annotations)
            filename = str(letterClass) + "_" + str(labelIDX) + "_" +  str(n) + ".png"
            output_image_path = os.path.join(folderPath, filename)
            n += 1

            cv2.imwrite(output_image_path, img)
            letterClassCounts[letterClass] += 1
        
        print(" ", letterClass, n)











""""
DATA AUGMENTATION OF TRAINING SET
"""

def all_subsets(ss):
    return list(chain(*map(lambda x: combinations(ss, x), range(1, len(ss)+1))))


def augment_training_set(training_set_folder, possible_transforms, number_of_augmentations):

    all_combinations = all_subsets(possible_transforms)
    print("number of augmentation method combinations:", len(all_combinations))


    for folder in tqdm(os.listdir(training_set_folder)):
        counter = 0
        #print(folder)

        folderPath = os.path.join(training_set_folder, folder)
        orginalFiles = [os.path.join(folderPath, filename) for filename in os.listdir(folderPath)]

        for orig_img_path in orginalFiles:
            counter += 1
            orig_img = cv2.imread(orig_img_path, cv2.IMREAD_GRAYSCALE)

            # for each possible combination of aug methods
            for subset in random.sample(all_combinations, number_of_augmentations):  
                img = orig_img.copy()

                transformations = list(subset)
                random.shuffle(transformations) # do the transformations in a random order
                fileName = folder + "_" + str(counter) + "_"
                
                for transform in transformations:
                    #print(transform, transform[0])
                    fileName += transform[0]

                    if transform == "rotate":
                        img = imageRotator(img, rotation_range=15) # rotate up to 15 degrees either direction
                        imageDilator, imageEroder
                    elif transform == "shear":
                        img = imageShearer(img, shear_range=0.15)
                    elif transform == "warp":
                        img = letterImageWarper(img)
                    elif transform == "erode":
                        img = imageEroder(img, max_erode_size=4)    # erode and dilates with a kernel of a random size in range [2, max_X_size]
                    elif transform == "dilate":
                        img = imageDilator(img, max_dilate_size=4)

                # finalise the path of the augmented image, and save it
                fileName += ".png"
                target_path = os.path.join(folderPath, fileName)
                cv2.imwrite(target_path, img)
