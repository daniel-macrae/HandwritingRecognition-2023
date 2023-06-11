from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def collate_fn(batch):
    return tuple(zip(*batch))


def datasetSplitter(dataset, batch_size=1, validation_split=0.2, shuffle=True, random_seed = 42):
    # set indicies of training and validation set
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating train and validation data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler, collate_fn=collate_fn)

    return train_loader, validation_loader

