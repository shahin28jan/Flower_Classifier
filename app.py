import streamlit as st
import tensorflow as tf
import streamlit as st

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('flower.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Flower Classification
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (180,180)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    #st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    argmax_index = np.argmax(predictions) # [0, 0]
    if argmax_index == 0:
        st.image(image, caption="predicted: daisy")
    elif argmax_index == 1:
        st.image(image, caption="predicted: dandelion")
    elif argmax_index == 2:
        st.image(image, caption="predicted: rose")
    elif argmax_index == 3:
        st.image(image, caption="predicted: sunflower")
    else:
        st.image(image, caption="predicted: tulip")