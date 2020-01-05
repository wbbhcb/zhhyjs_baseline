# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 18:30:24 2020

@author: hcb
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

trn_path = 'hy_round1_train_20200102'
test_path = 'hy_round1_testA_20200102'


def transform_to_picture(path, save_base_path):
    if not os.path.exists(save_base_path):
        # print(1)
        os.mkdir(save_base_path)
    for file in tqdm(os.listdir(path)):
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        name = file.split('.')[0]
        save_path = os.path.join(save_base_path, name + '.jpg')

        plt.figure(figsize=(4, 4))
        plt.plot(df['x'].values, df['y'].values)
        plt.axis('off')
        plt.savefig(save_path)
    #         plt.show()


transform_to_picture(trn_path, 'train')
transform_to_picture(test_path, 'test')