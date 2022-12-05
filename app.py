import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import base64
class_names=[
  'Penyakit Bacterial spot',
  'Penyakit Early blight',
  'Penyakit Late blight',
  'Penyakit Leaf Mold',
  'Penyakit Septoria leaf spot',
  'Penyakit Spider mites',
  'Penyakit Target Spot',
  'Penyakit Yellow Leaf Curl Virus',
  'Penyakit mosaic virus',
  'Sehat',]
penjelasan=[
'https://plantix.net/id/library/plant-diseases/300050/bacterial-spot-and-speck-of-tomato',
'https://plantix.net/id/library/plant-diseases/100321/early-blight',
'https://plantix.net/id/library/plant-diseases/100046/tomato-late-blight',
'https://plantix.net/id/library/plant-diseases/100257/leaf-mold-of-tomato',
'https://plantix.net/id/library/plant-diseases/100152/septoria-leaf-spot',
'https://plantix.net/id/library/plant-diseases/500004/spider-mites',
'https://apps.lucidcentral.org/pppw_v10/text/web_full/entities/tomato_target_spot_163.htm',
'https://plantix.net/id/library/plant-diseases/200036/tomato-yellow-leaf-curl-virus',
'https://plantix.net/id/library/plant-diseases/200037/tobacco-mosaic-virus',
'https://plantix.net/id/library/crops/tomato']

# background image to streamlit

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('white-triangle-pattern-seamless-background-2.jpg')

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('vgg16categori.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Upload', 'Gallery'], 
        icons=['house', 'cloud-upload'], menu_icon="cast", default_index=1)

if selected == "Home":
  col1, col2, col3 = st.columns(3)
  with col1:
    st.write(' ')
  with col2:
    st.image("20180911125756.png", width=200,use_column_width=True)
  with col3:
    st.write(' ')
  st.markdown("## Klasifikasi Penyakit Tanaman Tomat")
  st.markdown("""
ini adalah aplikasi deep learning (Convolutional Neural Network) untuk melakukan klasifikasi penyakit tanaman tomat, berikut adalah penyakit yang di klasifikasi:
1. Bacterial spot
2. Early blight 
3. Late blight
4. Leaf mold
5. Septoria leaf spot
6. Spider mite
7. Target spot
8. Mosaic virues
9. Yellow leaf virus""")
  st.image('image1.jpg', width= 730,use_column_width=True)

elif selected == "Upload":
  st.subheader("klasifikasi penyakit tanaman tomat")
  file = st.file_uploader("Silahkan upload gambar daun tomat" ,type=["jpg","png","jpeg"])
  import cv2
  from PIL import Image, ImageOps
  st.set_option('deprecation.showfileUploaderEncoding', False)
  def import_and_predict(image_data, model):
        size = (256,256)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
               
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        return prediction

  if file is None:
    st.text("silahkan upload gambar")
  else:
    image = Image.open(file)
    st.image(image, width=150)
    predictions = import_and_predict(image, model)
    score=np.array(predictions[0])
    st.text(
    "Tanaman Tomat Anda Terdeteksi {} dengan {:.2f} persentase."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    st.markdown("Informasi Penyakit : {}".format(penjelasan[np.argmax(score)]) ) 

elif selected == "Gallery":
  st.header("Galeri gambar")
  st.subheader("sample dataset")
  st.markdown("Data source: https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf")
  col1, col2, col3 = st.columns(3)
  with col1:
    st.image("images0.jpg", width=100)
  with col2:
    st.image("images1.jpg", width=100)
  with col3:
    st.image("images2.jpg", width=100)

  col1, col2, col3 = st.columns(3)
  with col1:
    st.image("images3.jpg", width=100)
  with col2:
    st.image("images4.jpg", width=100)
  with col3:
    st.image("images5.jpg", width=100)

  col1, col2, col3 = st.columns(3)
  with col1:
    st.image("images6.jpg", width=100)
  with col2:
    st.image("images7.jpg", width=100)
  with col3:
    st.image("images8.jpg", width=100)
