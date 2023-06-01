import torch.nn as nn


class CharacterCNN(nn.Module):
    def __init__(self, numChannels = 1, classes = 27, batch_size = 16, dropout_rate = 0.5):
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
        self.conv2 = nn.Conv2d(32, batch_size, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.fc1 = nn.Linear( batch_size *12* 12, 64)  # 50x 50 input
        self.fc1 = nn.Linear( batch_size *6* 6, 64)  # 32x 32 input

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
    def __init__(self, num_classes=27, batch_size = 16, dropout_rate = 0.5):
        super(DanNet1, self).__init__() 
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, batch_size, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(batch_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        #self.fc = nn.Linear(batch_size *24 *24, 128) # 50x50 input
        self.fc = nn.Linear(batch_size *12 *12, 128) # 32x32 input
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        return out


class LeNet5(nn.Module):
    def __init__(self, num_classes=27, batch_size = 16, dropout_rate = 0.5):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, batch_size, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(batch_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(batch_size * 5 * 5, 120)
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