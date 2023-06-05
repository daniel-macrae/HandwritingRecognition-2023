from classification_models.CNN_models import CharacterCNN, LeNet5, DanNet1
import torch
import torchvision.transforms as transforms
from data_management.augmentation.commonAug import imgResizer
from classification_models.Hebrew_Classes import hebrewCharacters
import torch



def get_dss_classifier_model(path_to_saved_CNN, device):

    model = LeNet5()#.to(device)\
    model.load_state_dict(torch.load(path_to_saved_CNN, map_location=device))  # map_location makes it possible to load the model trained on a gpu, onto a cpu (for eval)
    model.to(device)

    model.eval()

    return model


def classify_letters(image, BB_groups_sorted, model, device):
    transform = transforms.ToTensor()
    character_names = list(hebrewCharacters.keys())

    classified_text = [] # list of strings, one string of letters per row in the image
    for row in BB_groups_sorted:
        row_letters = ""
        for bb in row: # for letter BB in this row
            # crop out the letter and resize it
            x1,y1,x2,y2 = bb
            letterIM = image[y1:y2, x1:x2] 
            letterIM = imgResizer(letterIM, desired_size=32)

            tensor = torch.unsqueeze(transform(letterIM), 0) # convert letter img to tensor (1,1,32,32)
            tensor = tensor.to(device)

            output = model.forward(tensor)                   # run classifier
            predicted_class = torch.argmax(output, 1).item() # get the classifer result
            letter_name = character_names[predicted_class]   # find the class name
            hebrew_letter = hebrewCharacters[letter_name]    # get the ascii hebrew character
            row_letters += hebrew_letter                     # add new letter to the row

        classified_text.append(row_letters)

    return classified_text