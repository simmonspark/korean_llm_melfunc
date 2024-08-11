import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import *


class Korean_Dataset(Dataset):
    def __init__(self, X_encoded, labels, mode = 'train'):

        self.context = X_encoded
        self.labels = labels
        self.len = len(labels)
        self.mode = mode
    def __getitem__(self, item):
        context_enc = toknizing(self.context[item],self.mode)
        mask = context_enc['attention_mask']
        context_enc = context_enc['input_ids']

        labels_enc = toknizing(self.labels[item],self.mode)
        labels_enc = labels_enc['input_ids']
        return {'input_ids': context_enc.squeeze(0), 'labels': labels_enc.squeeze(0), 'attention_mask' : mask}
    def __len__(self):
        return self.len

if __name__ == "__main__":

    train_x, test_x, train_y, test_y = preprocessing(get_data(data_dir))

    dataset = Korean_Dataset(train_x, train_y)
    loader = DataLoader(dataset,batch_size=32)
    data = next(iter(loader))
