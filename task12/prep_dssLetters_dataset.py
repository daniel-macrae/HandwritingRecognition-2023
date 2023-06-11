import os
from data_management.dssLetters import duplicates_remover, train_test_val_splitter, fillInImbalancedClasses
import shutil
from data_management.dssLetters import augment_training_set


"""
SETTINGS
"""




source_folder = "monkbrill2/"           # where the original data is
temp_folder = "Data/dssLettersTemp"     # for intermediate operations, gets deleted when this code is done
target_folder = "Data/dssLetters"       # where to put the final train,test,validate folders


# thresholds for removing 'duplicate' images
positionThreshold = 50 
sizeThreshold = 5

# make test, train, (validation) set folders into the target_folder
validation_set = False  # whether or not we want a seperate validation set (makes the training set smaller)

reshape = True         # if we want to reshape all of the images now
desired_size = 32      # and to what size (eg. 50 is 50x50)


# whether or not to fill in the class imbalance with Habbakuk images
fillClassImbalance = True
fill_amount = 0.3 # up to 30% of the class with the max number of training samples



augment_training = True  # yes or no augmentaion applied on the whole training set
possible_transforms = ["rotate", "shear", "warp", "erode", "dilate"]
number_of_augmentations = 10   # 5 sets of augmentions per original image




"""
RUN STUFF
"""

if __name__ == "__main__":

    ######################################################
    # make the folders

    os.makedirs(temp_folder, exist_ok = True)
    training_set_folder = target_folder + "/train"

    try:   # delete the training data folder, if its already there (basically overwrite whats there)
        shutil.rmtree(target_folder)   
        os.makedirs(target_folder, exist_ok = True)
    except:
        os.makedirs(target_folder, exist_ok = True)


    ######################################################
    # DUPLICATES REMOVAL


    # ignore images that have x AND y positions that are less different (to any other image from the same DSS) 
    # than positionThreshold, AND same for width AND height for sizeThreshold

    print("\ncopying non-duplicate images to a temporary folder")
    duplicates_remover(source_folder, temp_folder, positionThreshold, sizeThreshold)



    ######################################################
    # make train, test, validation sets


    print("\nmaking train, test, (& validation) folders")
    train_test_val_splitter(temp_folder, target_folder, validation_set = validation_set, reshape = True, desired_size = desired_size)
    # also deletes the temp_folder


    ######################################################
    # using Habbakuk letter images to address class imbalance

    if fillClassImbalance:
        # fill in class imbalances using the Habbakuk letters
        
        # fills in each class with new images until it has the same number as the class with the most images
        # could be less than 1 (e.g. 0.7 to prevent having too many Habbakuk letters relative to the real DSS letters)
        # or more than 1 (e.g. adds habbakuk letters to all classes)

        print("\naddressing class imbalance by generating new images")
        fillInImbalancedClasses(training_set_folder, amount=fill_amount, image_size=desired_size)




    ######################################################
    # data augmentation 

    if augment_training:
        print("\naugmenting the training set")
        print("methods:", possible_transforms)
        augment_training_set(training_set_folder, possible_transforms, number_of_augmentations)
    print("DONE!")