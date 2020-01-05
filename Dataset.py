from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
import os
import pandas as pd


class MyDataset(Dataset):

    def __init__(self, data_path, figure_path, is_TrainData=True):
        df = pd.DataFrame()
        df['filename'] = os.listdir(data_path)
        df['id'] = df['filename'].apply(lambda x: int(x.split('.')[0]))
        df = df.sort_values('id', ascending=True).reset_index()
        self.data_path = data_path
        self.figure_path = figure_path
        self.is_TrainData = is_TrainData
        self.df = df
        # print(self.df.head())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df['filename'][idx]
        filepath = os.path.join(self.data_path, filename)
        figurepath = os.path.join(self.figure_path, str(self.df['id'][idx]) + '.jpg')
        img = cv2.imread(figurepath)
        # 转换成灰度图
        img = cv2.resize(img, (224, 224))
#        img = np.transpose(img, (1, 2, 0))
        img = np.array(img/255)
#        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#        img = np.reshape(img, (3, img.shape[1], img.shape[2])) / 255
        # print(filename)
        x = torch.tensor(img, dtype=torch.float)
        x = x.permute(2,0,1)
        if self.is_TrainData:
            data = pd.read_csv(filepath)
            target = data['type'][0]
            if target == '拖网':
                target = 0
            elif target == '刺网':
                target = 1
            else:
                target = 2
            target = torch.tensor(target, dtype=torch.long)

            return x, target
        else:
            return x

