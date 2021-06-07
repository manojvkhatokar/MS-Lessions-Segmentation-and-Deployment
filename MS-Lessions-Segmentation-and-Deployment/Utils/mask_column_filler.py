import csv 
import os 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt

df = pd.read_csv('../datasets/data_paths.csv')


for i in range(0,len(df)):
    if(cv2.imread(df.mask_path[i]).max() == 255):
        df['mask'][i] = int(1)
    else:
        df['mask'][i] = int(0)

df.to_csv('../datasets/data_paths.csv', index = False)



