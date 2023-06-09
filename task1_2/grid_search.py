import torch
import random
import pandas as pd
from sklearn.model_selection import ParameterGrid
from classification_models import CNN_models
import classifier
import argparse
import os

import traceback


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
 
    parser.add_argument("--model", default="LeNet5", type=str, metavar="N", help="name of the CNN model")

    ## POSSIBLE NAMES OF CNNs ARE:
    # LeNet5
    # DanNet1
    # CharacterCNN
    
    parser.add_argument("--epochs", default=50, type=int, help="number of epochs to run")
    parser.add_argument("--filename", default="grid_search", type=str, help="name of file to store the grid search results in")

    return parser



def grid_search(args):
    print("Starting Grid Search")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = [16, 32, 64, 128]
    LR = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    DROPOUT_RATE = [0, 0.2, 0.5]
    EPOCHS = args.epochs


    hyper_grid = {'batch_size' : BATCH_SIZE,
                'learning_rate' : LR,
                'dropout_rate' : DROPOUT_RATE}

    grid = list(ParameterGrid(hyper_grid))
    random.shuffle(grid)  # randomly shuffle the grid (in case we don't get many trials done, at least there is more variety)
    print("Size of grid:", len(grid))


    # AMOUNT OF GRID TO SAMPLE
    gridSampleSize = 0.7
    param_columns = list(grid[0].keys())
    
 
    output_filename = str(args.filename) + ".xlsx"
    saving_dir = os.path.join("classification_models", args.filename)
    os.makedirs(saving_dir, exist_ok = True)
    saving_file = os.path.join(saving_dir, output_filename)



    # GRID SEARCH LOOP
   
    RESULTS_DATAFRAME = pd.DataFrame(columns=[  "CNN_model",'batch_size', 'learning_rate', 'dropout_rate',"train_loss", "validadtion_loss",
                                                "train_accuracy", "validation_accuracy", 
                                                "lowest_validation_loss" , "lowest_validation_loss_epoch", "highest_validation_accuracy" , "highest_validation_accuracy_epoch",
                                                "time_to_train", "epochs"])

    print("running grid search for ", str(args.model))
    idx = 0
    for params in grid:    

        # random sampling...
        if random.random() > gridSampleSize:  # samples the grid
            continue # skips this set of parameters
        
        # check to see if these parameters have already been tried
        try:
            df = pd.read_excel(saving_file)
            if (df[param_columns] == params).all(1).any():
                continue
        except: pass

        # for the results of this episode (and set of parameters)
        
        if  args.model ==  "LeNet5":
          CNN_model = CNN_models.LeNet5(dropout_rate=params['dropout_rate'])
        elif args.model ==  "DanNet1":
          CNN_model = CNN_models.DanNet1(dropout_rate=params['dropout_rate'])
        else: 
          CNN_model = CNN_models.CharacterCNN(dropout_rate=params['dropout_rate'])
        
        
        idx += 1
        print("training model #", idx)
        time_to_train = None
        train_loss = None
        validation_loss = None
        train_accuracy = 0
        validation_accuracy = 0

        train_loss, train_accuracy, validation_loss, validation_accuracy, lowestValLoss, lowestValLossEpoch, highestValAccuracy, highestValAccuracyEpoch, time_to_train, total_epochs = classifier.trainModel(
            CNN_model, args, INIT_LR=params['learning_rate'], BATCH_SIZE=params['batch_size'],
            DROPOUT_RATE=params['dropout_rate'], gridsearch = True)

        # store the results in a dataframe, making a new row for this trial here
        tempDict = {"CNN_model" : args.model, 
                    "train_loss" : train_loss, 
                    "validation_loss" : validation_loss,
                    "train_accuracy"  : train_accuracy,
                    "validation_accuracy" : validation_accuracy,
                    "lowest_validation_loss" : lowestValLoss,
                    "lowest_validation_loss_epoch" : lowestValLossEpoch, 
                    "highest_validation_accuracy" : highestValAccuracy, 
                    "highest_validation_accuracy_epoch" : highestValAccuracyEpoch,
                    "time_to_train" : time_to_train,
                    "epochs" : total_epochs}
        
        resultsDict = {**params.copy(), **tempDict}  # make a line for in the results dict


        # Let parallel runs write to the same results file
        """
        try:
            RESULTS_DATAFRAME = pd.read_excel(output_filename)
            RESULTS_DATAFRAME.loc[len(RESULTS_DATAFRAME)+1] = resultsDict
        except:
            RESULTS_DATAFRAME.loc[0] = resultsDict
            #output_filename = 'grid_backup.xlsx'
        """
        
        try:
            RESULTS_DATAFRAME = pd.read_excel(saving_file)
            RESULTS_DATAFRAME.loc[len(RESULTS_DATAFRAME)+1] = resultsDict
        except Exception:
            traceback.print_exc()
            RESULTS_DATAFRAME = pd.DataFrame(resultsDict, index=[0])
        
        RESULTS_DATAFRAME.drop(RESULTS_DATAFRAME.filter(regex="Unnamed"), axis=1, inplace=True)
        RESULTS_DATAFRAME.to_excel(saving_file, index=False) # saves on every iteration (in case this takes long, or crashes, we can still pull the results out)

        
if __name__ == '__main__':
    args = get_args_parser().parse_args()
    grid_search(args)

