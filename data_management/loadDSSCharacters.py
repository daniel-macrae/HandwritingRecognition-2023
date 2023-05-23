from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os



class dssLettersDataset(Dataset):
    def __init__(self, folder_path):
        self.root = folder_path
        self.data = self.parseFolders()  # self.data is a list of tuples, which contain (pathToImage, imageClassLabel)

        self.convert_tensor = transforms.ToTensor()
        self.characterClasses = os.listdir(self.root)

        self.resize_image = transforms.Resize((50,50), antialias=True)

    def parseFolders(self): 
        dataTuples = []
        for idx, folderName in enumerate(os.listdir(self.root)):
            # path to the images in the folders
            imageFolderPath = os.path.join(self.root, folderName)
            
            label = idx # classification label (index of the classes)
            #imageFiles = [f for f in os.listdir(imageFolderPath) if os.path.isfile(os.path.join(imageFolderPath, f))]
            
            for imgFilename in os.listdir(imageFolderPath):
                
                # path to each image file
                imgPath = os.path.join(imageFolderPath, imgFilename)
                if os.path.isdir(imgPath):
                    # skip directories
                    continue
                dataTuples.append( (imgPath,label) ) # tuple of (path to image, classification label)

        return dataTuples

    def __getitem__(self, index):
        path, label = self.data[index]
        img = Image.open(path)
        
        input_img = self.convert_tensor(img)
        input_img = self.resize_image(input_img)  # RESIZING?

        #print(input_img.shape)

        return input_img, label
    
    def __len__(self):
        return len(self.data)