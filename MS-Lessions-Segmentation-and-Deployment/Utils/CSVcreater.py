import csv
import os

masks_array = []
images_array = []

def fillArray(masks_dir, images_dir): 

    for path in os.listdir(masks_dir):
        full_path_masks = os.path.join(masks_dir,path)
        if os.path.isfile(full_path_masks):
            masks_array.append(full_path_masks)            

    for path in os.listdir(images_dir):
        full_path_images = os.path.join(images_dir,path)
        if os.path.isfile(full_path_images):
            images_array.append(full_path_images)

def fillCsv( patient_id,dir_size):

    with open('../datasets/data_paths.csv', mode='a') as data:
        writer = csv.writer(data)
        for i in range(0,dir_size):
            writer.writerow([patient_id,images_array[i],masks_array[i]])

path1 = "D:/brainStripped/alljpgmasks/case10masks"
path2 = "D:/brainStripped/alljpgimages/case10images"
fillArray(path1,path2)
fillCsv("case10",270)




    

