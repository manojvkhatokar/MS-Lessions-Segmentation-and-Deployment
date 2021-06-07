#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, glob
import nibabel as nib
import dicom2nifti

# use numpy version - 1.17.5
import numpy as np
import cv2
dicom2nifti.convert_directory('D:/case7dicom', 
                              'D:/case7nifti', 
                              compression = False, 
                              reorient = True)


# In[3]:


import os
import imageio
input_dir = 'D:/case7nifti/'
output_dir = 'D:/case7jpgs/'

input_image_list = os.listdir(input_dir)

for f in input_image_list:
    img_path = os.path.join(input_dir, f)
    img = nib.load(img_path)
    img_fdata = img.get_fdata()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

#     print(img.shape)

    (x,y,z) = img.shape
    for i in range(z):
        slicing = img_fdata[:, :, i]
        imageio.imwrite(os.path.join(output_dir,'{}.jpg'.format(str(i).zfill(4))), slicing)


# In[33]:


import os
import imageio
input_dir = 'D:/WORK/PhD/start/MSProject/datasets/rebecca/nifti_masks/'
output_dir = 'D:/WORK/PhD/start/MSProject/datasets/rebecca/jpg_masks_y/'

input_image_list = os.listdir(input_dir)

for f in input_image_list:
    img_path = os.path.join(input_dir, f)
    img = nib.load(img_path)
    img_fdata = img.get_fdata()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

#     print(img.shape)

    (x,y,z) = img.shape
    for i in range(y):
        slicing = img_fdata[:, i, :]
        imageio.imwrite(os.path.join(output_dir,'{}_mask.jpg'.format(str(i).zfill(4))), slicing)


# In[32]:


import os
import imageio
input_dir = 'D:/WORK/PhD/start/MSProject/datasets/rebecca/nifti_masks/'
output_dir = 'D:/WORK/PhD/start/MSProject/datasets/rebecca/jpg_masks_x/'

input_image_list = os.listdir(input_dir)

for f in input_image_list:
    img_path = os.path.join(input_dir, f)
    img = nib.load(img_path)
    img_fdata = img.get_fdata()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

#     print(img.shape)

    (x,y,z) = img.shape
    for i in range(x):
        slicing = img_fdata[i, :, :]
        imageio.imwrite(os.path.join(output_dir,'{}_mask.jpg'.format(str(i).zfill(4))), slicing)


# In[ ]:




