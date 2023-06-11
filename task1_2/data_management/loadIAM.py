from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os



class IAMDataset(Dataset):
    def __init__(self, folder_path):
        self.root = folder_path
        self.data = self.parseAnnotationsFile()  # self.data is a list of tuples, which contain (image filename, image label)

        self.imageFolder = os.path.join(self.root, 'img')

        self.convert_tensor = transforms.ToTensor()

    def parseAnnotationsFile(self): 
        pathToTxtFile = os.path.join(self.root, 'iam_lines_gt.txt')

        with open(pathToTxtFile, 'r') as infile:
            lines = [line[:-1] for line in infile]   # the [:-1] removes the '/n' in the strings

            # loop through the annotations, three items at a time
            lines = list(zip(*[iter(lines)]*3))
            linesTuples = [(line[:-1]) for line in lines] # drop the empty space

        return linesTuples

    def __getitem__(self, index):
        filename, label = self.data[index] # contains filename, label (label is the text thats in the image)

        img = Image.open(os.path.join(self.imageFolder, filename))
        input_img = self.convert_tensor(img)

        return input_img, label
    
    def __len__(self):
        return len(self.data)