import sys
import zipfile
import pandas as pd
import os
import torch
from tqdm import tqdm
import csv
import requests
from sklearn.model_selection import train_test_split
import zipfile
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor, ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
from datasets import load_dataset
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from torch.utils.data import DataLoader
from datasets import load_metric

# downloads file from a URL
def download_file(url, save_path):
    print(f"Downloading File at {save_path}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for non-2xx status codes
        with open(save_path, "wb") as file:
            file.write(response.content)
        print("Download completed!")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the file: {e}")

# creates a dataframe compatible with out dataset builder class
def create_df(images_path):
    files = os.listdir(images_path)
    df = pd.DataFrame(files, columns=['file_name'])
    df["text"] = ""
    return df

# creates a dataset out of a dataframe
class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels), "file_name": file_name}
        return encoding
    
    
def main(args):
    # check if argument exists
    if(len(args) != 2):
        print('Images directory argument is missing')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    current_directory = os.getcwd()
    image_folder = args[1]
    
    
    # check if image folder exists
    if(len(image_folder) < 1):
        print("Invalid folder argument")
        return
    if(image_folder[-1] != '/'):
        image_folder = image_folder + '/'
    
    print(os.path.join(current_directory, image_folder))
    
    if not os.path.exists(os.path.join(current_directory, image_folder)):
        print(f"The folder '{os.path.join(current_directory, image_folder)}' is not found.")
        return
    
    # create dataframe from images contained in a folder
    df = create_df(image_folder)
    df['length'] = df['text'].str.len()
    print(f"Found {df.shape[0]} files")

    # processor used in the dataset class to transform images, resize and normalize
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    # create a dataset out of the dataframe
    test_dataset = IAMDataset(root_dir=os.path.join(current_directory, image_folder),
                               df=df,
                               processor=processor)
    # create a dataloader to feed the evaluation in batches
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    # Checks if the model checkpoint exists, if not it downloads it and extracts its contents
    checkpoint = 'checkpoint-10000'
    save_location =  os.path.join(current_directory, f'{checkpoint}.zip')
    model_folder_location =  os.path.join(current_directory, f'{checkpoint}')
    
    if os.path.exists(model_folder_location):
        print(f"The model '{model_folder_location}' is already unzipped.")
    else:
        if os.path.exists(save_location):
            print(f"The file '{save_location}' exists.")
        else:
            print(f"The file '{save_location}' does not exist.")
            file_url = f"https://storage.googleapis.com/ayuda/{checkpoint}.zip"
            download_file(file_url, save_location)
        
        print('unzipping directory')
        model_zip = f"{checkpoint}.zip"
        img_zip_path = os.path.join(current_directory, model_zip)
        zip_ref = zipfile.ZipFile(img_zip_path, 'r')
        zip_ref.extractall(current_directory)
        zip_ref.close()
        print('unzipped directory!')

    # loads model checkpoint to device
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint)
    model.to(device)

    print("Generating text")

    # generate texts on batches
    results = []
    for batch in tqdm(test_dataloader):
        pixel_values = batch["pixel_values"].to(device)
        outputs = model.generate(pixel_values)
        pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
        results = results + [(file_name, pred) for file_name, pred in zip(batch["file_name"], pred_str) ]
    # craetes a results folder if it does not exists
    results_path = os.path.join(current_directory, 'results')
    if not os.path.exists(results_path):
        # Create the folder
        os.makedirs(results_path)
        print(f"The folder '{results_path}' was created.")
    else:
        print(f"The folder '{results_path}' already exists.")
    # writes a txt file for each image in which text was generated
    for img, pred in results:
        with open(os.path.join(current_directory, f'results/{img.replace(".png","")}_characters.txt'), 'w') as file:
            file.write(f"{pred}")


    print("Done, results written inside /results folder")
    return

if __name__ == "__main__":
    args = sys.argv
    main(args)
