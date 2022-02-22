# import libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from tensorflow import keras
from PIL import Image
import cv2 as cv
import numpy as np

# Displaying in-head
st.title("Quelle race de chien ?")
st.text("Charger une image pour connaitre sa race.")

# Importing labels name
my_content = open("dogs_name.txt", "r")
dog_names = my_content.read()
dogs_list = dog_names.split('\n')
my_content.close()

def image_classifier(img, weights_file):
  """Fonction pour définir la prédiction
  """
  model = keras.models.load_model(weights_file)
  
  
  # Convert to RGB
  img = cv.cvtColor(np.float32(img),cv.COLOR_BGR2RGB)
  # Resize image
  dim = (299, 299)
  img = cv.resize(img, dim, interpolation=cv.INTER_LINEAR)
  # Equalization
  img_yuv = cv.cvtColor(img,cv.COLOR_BGR2YUV)
  img_yuv[:,:,0] = cv.equalizeHist(np.uint8(img_yuv[:,:,0]))
  img_equ = cv.cvtColor(img_yuv, cv.COLOR_YUV2RGB)
  # Apply non-local means filter on test img
  dst_img = cv.fastNlMeansDenoisingColored(
    src=np.uint8(img_equ),
    dst=None,
    h=10,
    hColor=10,
    templateWindowSize=7,
    searchWindowSize=21)
  
  #Convert modified img to array
  img_array = keras.preprocessing.image.img_to_array(img_equ)
    
  # Apply preprocess Resnet
  img_array = img_array.reshape((-1, 299, 299, 3))
  img_array = tf.keras.applications.resnet.preprocess_input(img_array)
    
  # Predictions
  predictions = model.predict(img_array).flatten()
  
  predictions = tf.nn.softmax(predictions)
  predictions = np.argmax(predictions)
  return dogs_list[predictions]

# Displaying uploader
uploaded_file = st.file_uploader("Chargement de l'image...", type="jpg")

# Loop ending with prediction
if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img, caption='Image chargee.', use_column_width=True)
  st.write("")
  st.write("Classification")
  label = image_classifier(img, 'monModele.h5')
  st.write(label)


st.markdown("*Created by Jérôme Walroff.*")