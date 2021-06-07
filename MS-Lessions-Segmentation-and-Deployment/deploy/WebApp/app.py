import streamlit as st
import matplotlib.pyplot as plt 
import cv2
import numpy as np
from keras.models import model_from_json
import os

import tensorflow as tf
from PIL import Image
import numpy as np
from skimage import transform
import glob


def main():
    st.title("Automated Detection of Multiple Sclerosis Lesions In Normal Appearing White Matter From Brain MRI")    
    st.markdown("- Manoj V Khatokar 1DS17CS063 \n - M Hemanth Kumar 1DS17CS059 \n - Chandrahas Kuridi 1DS17CS050 \n - GUIDED BY : Prof. Shwetha M D")
    st.title("Results")

    def load(filename):
        np_image = Image.open(filename)
        np_image = np.array(np_image).astype('float32')/255
        np_image = transform.resize(np_image, (256, 256, 3))
        np_image = np.expand_dims(np_image, axis=0)
        return np_image

    def load_segmentation_model(path):
        seg_model = model_from_json(open('../../pretrained_models/Segmentation Review Model/modelP.json').read())
        seg_model.load_weights(os.path.join(os.path.dirname('../../pretrained_models/Segmentation Review Model/modelP.json'), 'modelWeights.h5'))
        seg_model.compile()
        ## segmentation of the lesions 
        user_image = cv2.imread(path)
        resized_img = cv2.resize(user_image,  (256, 192), interpolation = cv2.INTER_CUBIC)
        resized_img = resized_img / 255
        single_picture_prediction = seg_model.predict(np.array([resized_img,]))
        single_picture_prediction = np.round(single_picture_prediction, 0)
        plt.title('Predicted Segmentation')
        plt.imshow(resized_img)
        plt.imshow(single_picture_prediction.reshape(single_picture_prediction.shape[1], single_picture_prediction.shape[2]), alpha=0.4)
        st.text('Proven case of MS')
        st.pyplot(plt)

    st.sidebar.title("Segmentation Parameters")
    
        
          
    st.sidebar.subheader("Choose Segmentation Model")
    seg_model = st.sidebar.selectbox("Segmentation Model", ("DC Unet",))
    
    if seg_model == "DC Unet":
        st.sidebar.subheader("Enter the URL for the image")
        path = st.sidebar.text_input('URL')


        user_input = load(path)
        with open('../../pretrained_models/Classifier Review Model/classifier-resnet-model-for-MS-lesions-with-6epochs.json', 'r') as json_file:
            json_savedModel= json_file.read()
        classifier_model = tf.keras.models.model_from_json(json_savedModel)
        classifier_model.load_weights('../../pretrained_models/Classifier Review Model/classifier-resnet-weights-refined-6epochs.hdf5')
        classifier_model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ["accuracy"])
        predicted_for_user_input = classifier_model.predict(user_input)
        prediction_for_user_input = []
        prediction_for_user_input.append(str(np.argmax(predicted_for_user_input)))
        prediction_for_user_input = np.asarray(prediction_for_user_input)

        print(prediction_for_user_input)
        if(prediction_for_user_input[0]=='1'):
            load_segmentation_model(path)
        else:
            st.text('No MS Lesions found in this case')

            

        original_image = cv2.imread(path)
        original_image = cv2.resize(original_image,  (256, 192), interpolation = cv2.INTER_CUBIC)
        plt.title('Original Image')
        plt.imshow(original_image)
        st.sidebar.pyplot(plt)

        
        # if st.sidebar.button("Start Segmentation", key = 'start'):
        #     load_segmentation_model(path)

if __name__ == '__main__':
    main()
