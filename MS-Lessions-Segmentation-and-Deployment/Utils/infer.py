import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
import cv2
import numpy as np
from keras.models import model_from_json
import os
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd

## Loading the classifier model first 

with open('pretrained_models/Classifier Review Model/classifier-resnet-model-for-MS-lesions-with-6epochs.json', 'r') as json_file:
    json_savedModel= json_file.read()

# load the model  
classifier_model = tf.keras.models.model_from_json(json_savedModel)
classifier_model.load_weights('pretrained_models/Classifier Review Model/classifier-resnet-weights-refined-6epochs.hdf5')
classifier_model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ["accuracy"])


## inferencing the image class for userTest1.jpg




# Create a data generator for test image
filenames = os.listdir('D:/WORK/PhD/start/MSProject/UserTests')
data = pd.DataFrame({'filename':filenames, 'mask':mask})
test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=data,
directory= './',
x_col='image_path',
y_col='mask',
batch_size=16,
shuffle=False,
class_mode='categorical',
target_size=(256,256))


classification = classifier_model.predict(test_generator,steps = 1, verbose=1)

predict = []

for i in classification:
  predict.append(str(np.argmax(i)))

predict = np.asarray(predict)

print(predict)

## loading the segmentation model if classification gives positive for userTest1.jpg
seg_model = model_from_json(open('pretrained_models/Segmentation Review Model/modelP.json').read())
seg_model.load_weights(os.path.join(os.path.dirname('pretrained_models/Segmentation Review Model/modelP.json'), 'modelWeights.h5'))
seg_model.compile()
## segmentation of the lesions 
user_image = cv2.imread('UserTests/userTest3.jpg')
resized_img = cv2.resize(user_image,  (256, 192), interpolation = cv2.INTER_CUBIC)
resized_img = resized_img / 255
single_picture_prediction = seg_model.predict(np.array([resized_img,]))
single_picture_prediction = np.round(single_picture_prediction, 0)
plt.title('Predicted Lesion')
plt.imshow(single_picture_prediction.reshape(single_picture_prediction.shape[1], single_picture_prediction.shape[2]))
