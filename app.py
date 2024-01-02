import streamlit as st
import tensorflow as tf
st.set_option('depreciation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation = True)
def load_model():
    model = tf.keras.models.load_model('/kaggle/working/my_model.hdf5')
    return model
model = load_model()
st.write('''
    #Lost and Found Pets (Cats | Dogs)
    '''
        )
file = st.file_uploader("Please upload the picture of the pet found", type=[jpg, png, jpeg,])
import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    size = (256,256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asrray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width = True)
    predictions = import_and_predict(image, model)
    class_names = ['Dog', 'Cat']
    string = "This is an image of a "+class_names[np.argmax(predictions)]
    st.success(string)
