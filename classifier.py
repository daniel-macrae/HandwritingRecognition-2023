import matplotlib
# import the necessary packages
from sklearn.metrics import classification_report
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import torch
import torch.nn as nn


from data_management.loadDSSCharacters import dssLettersDataset

from classification_models import CNN_models






def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
 
    parser.add_argument("--model", default="LeNet5", type=str, metavar="N", help="name of the CNN model")

    ## POSSIBLE NAMES OF CNNs ARE:
    # LeNet5
    # DanNet1
    # CharacterCNN
    
    parser.add_argument("--epochs", default=50, type=int, help="number of epochs to run")
    
    parser.add_argument("--filename", default="classifier", type=str, help="name of file to store the results in")


    return parser



def trainModel(model, args, INIT_LR = 1e-3, BATCH_SIZE = 16, DROPOUT_RATE  = 0, gridsearch = False):

    train_dir = 'Data/dssLetters/train/'
    val_dir = 'Data/dssLetters/test/'

    train_set = dssLettersDataset(folder_path= train_dir)
    validation_set = dssLettersDataset(folder_path= val_dir)

    # define training hyperparameters
 
    EPOCHS = args.epochs

    VALIDATION_RATE = 5 # note down the train/val error every X epochs
    VERBOSE = False # Whether to print the losses on each validation step
    

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)



    # calculate steps per epoch for training and validation set
    trainSteps = len(train_loader.dataset) // BATCH_SIZE
    valSteps = len(validation_loader.dataset) // BATCH_SIZE

    # logging
    lowestValLoss = 1e10
    lowestValLossEpoch = None
    highestValAccuracy = 0
    highestValAccuracyEpoch = None


    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=INIT_LR)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss() #When we combine the nn.NLLoss class with LogSoftmax in our model definition, we arrive at categorical cross-entropy loss
    #  (which is the equivalent to training a model with an output Linear layer and an nn.CrossEntropyLoss loss).
    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    # measure how long training is going to take
    startTime = time.time()


    # loop over our epochs

    for e in range(1, EPOCHS + 1):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0
        # loop over the training set
        for (x, y) in train_loader:
            x = torch.stack([image.to(device) for image in x])
            y = torch.stack([torch.LongTensor([target]).to(device) for target in y])
            y = torch.squeeze(y)

            output = model(x)
            loss = criterion(output, y)

            # perform a forward pass and calculate the training loss
            
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (output.argmax(1) == y).type(
                torch.float).sum().item()

            
        if e % VALIDATION_RATE == 0:

            "EVALUATION"

            # switch off autograd for evaluation
            with torch.no_grad():
                # set the model in evaluation mode
                model.eval()
                
                # loop over the validation set
                for (x, y) in validation_loader:
                    x = torch.stack([image.to(device) for image in x])
                    y = torch.stack([torch.LongTensor([target]).to(device) for target in y])
                    y = torch.squeeze(y)
                    output = model(x)
                    
                    totalValLoss += criterion(output, y)
                    
                    # calculate the number of correct predictions
                    valCorrect += (output.argmax(1) == y).type(torch.float).sum().item()

            # calculate the average training and validation loss  #
            avgTrainLoss = (totalTrainLoss / trainSteps).item()  # and detatch from tensor (into a float, using .item())
            avgValLoss = (totalValLoss / valSteps).item()
            
            


            # calculate the training and validation accuracy
            trainCorrect = trainCorrect / len(train_loader.dataset)
            valCorrect = valCorrect / len(validation_loader.dataset)
            # update our training history
            H["train_loss"].append(avgTrainLoss)
            H["train_acc"].append(trainCorrect)
            H["val_loss"].append(avgValLoss)
            H["val_acc"].append(valCorrect)

            # logging
            if avgValLoss < lowestValLoss:
                lowestValLoss = avgValLoss
                lowestValLossEpoch = e + 1
            if valCorrect > highestValAccuracy:
                highestValAccuracy = valCorrect
                highestValAccuracyEpoch = e + 1

            # print the model training and validation information
            if VERBOSE:
                print("[INFO] EPOCH: {}/{}".format(e, EPOCHS))
                print("   Train loss: {:.6f}, Train accuracy: {:.4f}".format(
                    avgTrainLoss, trainCorrect))
                print("   Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
                    avgValLoss, valCorrect))
            
    print("gridsearch is ", gridsearch)
    if  gridsearch == True:
        return avgTrainLoss, trainCorrect, avgValLoss, valCorrect, lowestValLoss, lowestValLossEpoch, highestValAccuracy, highestValAccuracyEpoch, (time.time() - startTime)/60, EPOCHS 
    
    else:
        # only saving one for now
        saving_dir = os.path.join("classification_models/", args.filename)
        os.makedirs(saving_dir, exist_ok = True)

        # filenames that include the name of the model (e.g. LeNet5)
        saveArgs = {
            "plot_loss": saving_dir + "/plot_loss_" + args.model + "_bs_" + str(BATCH_SIZE) +"-LR_" + str(INIT_LR) + "_DR_" + str(DROPOUT_RATE) +" .png",
            "plot_acc": saving_dir + "/plot_acc_" + args.model + "_bs_" + str(BATCH_SIZE) +"-LR_" + str(INIT_LR) + "_DR_" + str(DROPOUT_RATE) +" .png",
            "model": saving_dir + "/model_" + args.model + "_bs_" + str(BATCH_SIZE) +"-LR_" + str(INIT_LR) + "_DR_" + str(DROPOUT_RATE) + ".pth"
        }
        # plot the training loss and accuracy
        epochs = np.arange(len(H["train_loss"]))*VALIDATION_RATE

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, H["train_loss"],  label="train_loss")
        plt.plot(epochs, H["val_loss"], label="val_loss")
        plt.title("Training and Validation Loss on Dataset" + str(args.model))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(saveArgs["plot_loss"])

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, H["train_acc"], label="train_acc")
        plt.plot(epochs, H["val_acc"], label="val_acc")
        plt.title("Training and Validation Accuracy on Dataset " + str(args.model))
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(saveArgs["plot_acc"])
        # serialize the model to disk
        torch.save(model.state_dict(), saveArgs["model"])

    print("DONE!")
    print("Duration: ", (time.time() - startTime)/60, "minutes")

    

if __name__ == "__main__":
    args = get_args_parser().parse_args()

 
    # load the model, using the string from args.model, from the CNN_models files
    print("Training model:", args.model)
    modelClassObject = getattr(CNN_models, args.model)
    model = modelClassObject()
    if str(args.model) == "LeNet5":
        LR = 0.00005
        DR = 0.2
        BS = 16
    elif str(args.model) == "DanNet1":
        LR = 0.00001
        DR = 0.2
        BS = 16
    else:
        LR = 0.0005     #0.005
        DR = 0.2
        BS = 64         #16

    trainModel(model, args, INIT_LR=LR, BATCH_SIZE=BS, DROPOUT_RATE=DR)