import cv2
import numpy as np # linear algebra
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import torch
import pickle


HEIGHT = 137
WIDTH = 236
DATA_DIR = '/home/chec/data/bengali'

def get_train():
    class GraphemeDataset(Dataset):
        def __init__(self, df):
            self.df = df
            self.image = self.df.iloc[:, 1:].values
            self.ids =  self.df.iloc[:, 0].values
        def __len__(self):
            return len(self.df)
        def __getitem__(self,idx):
            image = self.image[idx].reshape(137, 236).astype(np.uint8)
            id_ = self.ids[idx]

            np.save(f'{DATA_DIR}/train_images_npy/{id_}.npy', image)

            return torch.tensor(image)
    train = pd.read_csv(f'{DATA_DIR}/train.csv')
    data0 = pd.read_parquet(f'{DATA_DIR}/train_image_data_0.parquet')
    data1 = pd.read_parquet(f'{DATA_DIR}/train_image_data_1.parquet')
    data2 = pd.read_parquet(f'{DATA_DIR}/train_image_data_2.parquet')
    data3 = pd.read_parquet(f'{DATA_DIR}/train_image_data_3.parquet')

    data_full = pd.concat([data0,data1,data2,data3],ignore_index=True)
    test_image = GraphemeDataset(data_full)
    test_loader = torch.utils.data.DataLoader(test_image,batch_size=256, num_workers=8, shuffle=False)
    count = 0
    for data in tqdm(test_loader):
        count += 1
    print(count)

if __name__ == '__main__':
    get_train()
