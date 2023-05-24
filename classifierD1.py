import matplotlib
matplotlib.use("Agg") # ??¿¿

# import the necessary packages
from sklearn.metrics import classification_report
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import os
import torch
import torch.nn as nn

class CharacterCNN(nn.Module):
    def __init__(self, numChannels = 1, classes = 27, dropout_rate = 0.5):
        super(CharacterCNN, self).__init__()

        """
        self.conv1 = nn.Conv2d(numChannels, 32, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, BATCH_SIZE, kernel_size=5, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear( BATCH_SIZE *12* 12, 128) #this shoulf be BATCH_SIZE *12* 12 but that doesnt work
        self.fc2 = nn.Linear(128, classes)
        self.relu = nn.ReLU() # could also just use nn.functional.relu
        #self.logSoftmax = nn.LogSoftmax(dim=1)
        """
        self.conv1 = nn.Conv2d(numChannels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear( 16 *12* 12, 64) #this shoulf be BATCH_SIZE *12* 12 but that doesnt work
        self.fc2 = nn.Linear(64, classes)
        self.relu = nn.ReLU() # could also just use nn.functional.relu
        #self.logSoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout_rate)
 
    # Using nn.functional provides a more concise syntax since it directly applies the operations as functions,
    # while using the corresponding layers from torch.nn allows for explicit control over the layers used in the network
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        #x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1) #same as flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        #output = self.logSoftmax(output)
        return output


class DanNet1(nn.Module):
    def __init__(self, num_classes=27):
        super(DanNet1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(3600, 128)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        return out


class LeNet5(nn.Module):
    def __init__(self, num_classes=27):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

from data_management.loadDSSCharacters import dssLettersDataset



def trainModel():

    train_dir = 'Data/dssLetters/train/'
    val_dir = 'Data/dssLetters/test/'

    train_set = dssLettersDataset(folder_path= train_dir)
    validation_set = dssLettersDataset(folder_path= val_dir)

    # define training hyperparameters
    INIT_LR = 1e-3
    BATCH_SIZE = 128
    EPOCHS = 100

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)



    # calculate steps per epoch for training and validation set
    trainSteps = len(train_loader.dataset) // BATCH_SIZE
    valSteps = len(validation_loader.dataset) // BATCH_SIZE


    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DanNet1().to(device)
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

    for e in range(0, EPOCHS):
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

            
        if e % 5 == 0:

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

            # calculate the average training and validation loss
            avgTrainLoss = totalTrainLoss / trainSteps
            avgValLoss = totalValLoss / valSteps
            # calculate the training and validation accuracy
            trainCorrect = trainCorrect / len(train_loader.dataset)
            valCorrect = valCorrect / len(validation_loader.dataset)
            # update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["train_acc"].append(trainCorrect)
            H["val_loss"].append(avgValLoss.cpu().detach().numpy())
            H["val_acc"].append(valCorrect)

            # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
            print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
                avgTrainLoss, trainCorrect))
            print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
                avgValLoss, valCorrect))



    # only saving one for now
    os.makedirs("classification_models", exist_ok = True)

    args = {
        "plot_loss": "classification_models/plot_loss_D1.png",
        "plot_acc": "classification_models/plot_acc_D1.png",
        "model": "classification_models/model_D1.pth"
    }

    # plot the training loss and accuracy

    epochs = np.arange(len(H["train_loss"]))*10

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(epochs, H["train_loss"],  label="train_loss")
    plt.plot(epochs, H["val_loss"], label="val_loss")
    plt.title("Training and Validation Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(args["plot_loss"])

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(epochs, H["train_acc"], label="train_acc")
    plt.plot(epochs, H["val_acc"], label="val_acc")
    plt.title("Training and Validation Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(args["plot_acc"])
    # serialize the model to disk
    torch.save(model.state_dict(), args["model"])

    print("DONE!")
    print("Duration: ", (startTime - time.time())/60, "minutes")

if __name__ == "__main__":
    trainModel()