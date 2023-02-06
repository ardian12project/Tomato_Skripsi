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
'Bacterial Spot/ Bercak bakteri disebabkan oleh beberapa spesies bakteri dari genus Xanthomonas. Ini terjadi di seluruh dunia dan merupakan salah satu penyakit paling merusak pada tomat yang tumbuh di lingkungan yang hangat dan lembab. Patogen dapat bertahan hidup di dalam atau pada biji, pada puing-puing tanaman dan pada gulma tertentu. Bakteri ini memiliki masa bertahan hidup yang sangat terbatas dari hari ke minggu di tanah. Ketika kondisinya nyaman, bakteri menyebar melalui percikan air hujan atau irigasi pancur ke tanaman yang sehat. Memasuki jaringan tanaman melalui pori-pori dan luka pada daun. Kisaran suhu optimal untuk penyebarannya adalah 25 hingga 30 °C. Setelah tanaman terinfeksi, penyakit ini sangat sulit dikendalikan dan dapat menyebabkan kerugian total pada hasil panen.',
'Gejala-gejalanya disebabkan oleh Alternaria solani, jamur yang melewati musim dingin pada puing-puing tanaman yang terinfeksi di tanah atau pada inang alternatif. Bibit atau benih yang dibeli mungkin juga sudah terkontaminasi. Daun bagian bawah sering terinfeksi ketika kontak dengan tanah yang terkontaminasi. Suhu hangat (24-29 °C) dan kelembaban tinggi (90%) mendukung pengembangan penyakit ini. Periode basah yang panjang (atau cuaca basah/kering bergantian) meningkatkan produksi spora, yang dapat menyebar melalui angin, percikan air hujan atau irigasi pancur. Umbi yang dipanen masih hijau atau dalam kondisi basah sangat rentan terhadap infeksi. Jamur ini sering menyerang setelah periode curah hujan tinggi dan sangat merusak di daerah tropis dan subtropis.',
'Risiko infeksi paling tinggi terjadi pada pertengahan musim panas. Jamur memasuki tanaman melalui luka dan sobekan di kulit. Suhu dan kelembaban adalah faktor lingkungan terpenting yang mempengaruhi perkembangan penyakit. Jamur busuk daun tumbuh paling baik pada kelembaban relatif tinggi (sekitar 90%) dan dalam kisaran suhu 18 hingga 26 °C. Cuaca musim panas yang hangat dan kering dapat menghentikan penyebaran penyakit ini.',
'Gejala-gejalanya disebabkan oleh jamur Mycovellosiella fulva, yang sporanya dapat bertahan hidup tanpa inang selama 6 bulan hingga satu tahun pada suhu kamar (tidak harus). Kelembaban udara dan kelembaban daun yang berkepanjangan di atas 85% mendukung perkecambahan spora. Suhu antara 4 hingga 34 °C mendukung perkecambahan spora, dengan suhu optimalnya pada 24-26 °C. Kondisi kering dan ketiadaan air pada daun bisa merusak daya kecambah. Gejala biasanya mulai muncul 10 hari setelah inokulasi dengan perkembangan flek di kedua sisi daun. Di bagian bawah, sejumlah besar struktur penghasil spora terbentuk dan spora ini mudah menyebar dari tanaman ke tanaman dengan bantuan percikan air dan angin, peralatan dan pakaian pekerja, dan serangga. Patogen biasanya menginfeksi daun dengan menembus stomata pada tingkat kelembaban tinggi.',
'Bercak daun Septoria terjadi di seluruh dunia dan disebabkan oleh jamur Septoria lycopersici. Jamur hanya menginfeksi tanaman dari famili kentang dan tomat. Kisaran suhu untuk pengembangan jamur bervariasi antara 15 ° dan 27 °C, dengan pertumbuhan optimal pada 25 °C. Spora mungkin disebarkan oleh air dari irigasi curah, percikan air hujan, tangan dan pakaian pemetik, serangga seperti kumbang, dan peralatan budidaya. Jamur ini melewati musim dingin pada gulma solanaceous dan di tanah atau puing-puing tanah selama periode yang singkat.',
'Kerusakan disebabkan oleh tungau laba-laba dari genus Tetranychus, terutama T. urticae dan T. cinnabarinus. Betina dewasa memiliki panjang 0,6 mm, berwarna hijau pucat dengan dua bercak lebih gelap di tubuhnya yang oval dan rambutnya panjang di belakang. Pada musim dingin, betinanya berwarna kemerahan. Pada musim semi, betina bertelur bulat-bulat dan transparan di bagian bawah daun. Nimfa berwarna hijau pucat dengan tanda lebih gelap di sisi punggungnya. Tungau melindungi diri dalam kepompong di bagian bawah bilah daun. Tungau laba-laba tumbuh subur pada iklim kering dan panas dan akan berkembang biak hingga 7 generasi dalam satu tahun dalam kondisi ini. Ada berbagai macam inang alternatif, termasuk gulma.',
'Jamur Corynespora cassiicola bertahan melewati musim dingin pada sisa-sisa tanaman dan di tanah. Kondisi yang nyaman untuk infeksi adalah pada kelembaban tinggi (> 80%) dan kelembaban bebas pada daun. Cuaca kering akan menekan perkembangan penyakit. Penyakit ini berpotensi serius pada varietas yang matang lebih akhir atau pada varietas yang rentan selama curah hujan tinggi.',
'Virus Kuning Keriting Daun Tomat bukan berasal dari benih dan tidak ditularkan secara mekanis. Penyakit ini disebarkan oleh lalat putih dari spesies Bemisia tabaci. Lalat putih ini memakan permukaan daun bagian bawah dari sejumlah tanaman dan terpikat oleh tanaman muda yang lembut. Seluruh siklus infeksi dapat terjadi dalam waktu sekitar 24 jam dan didukung oleh cuaca kering dengan suhu tinggi',
'Virus ini dapat bertahan di sisa-sisa tanaman atau akar di tanah kering selama periode lebih dari 2 tahun (1 bulan di sebagian besar tanah). Tanaman bisa terkontaminasi melalui luka kecil di akar. Virus ini dapat menyebar melalui benih yang terinfeksi, bibit, gulma dan bagian tanaman yang terkontaminasi. Angin, hujan, belalang, mamalia kecil dan burung juga dapat membawa virus antar lahan. Praktik budidaya yang buruk selama penanganan tanaman juga mendukung penularan virus. Durasi waktu siang hari, suhu, dan intensitas cahaya serta varietas dan umur tanaman menentukan tingkat keparahan infeksi.',
'Teruskan perawatan anda agar tanaman anda selalu sehat']

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
ini adalah aplikasi deep learning (Convolutional Neural Network) untuk melakukan klasifikasi penyakit tanaman tomat, aplikasi ini hanya bisa mendeteksi penyakit pada daun tomat. berikut adalah penyakit yang di klasifikasi:
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
    st.subheader(
    "Tanaman Tomat Anda Terdeteksi {}."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    st.markdown("Informasi Penyakit :".format(penjelasan[np.argmax(score)]) )
    st.markdown("{}".format(penjelasan[np.argmax(score)]) )

elif selected == "Gallery":
  st.header("Galeri gambar")
  st.subheader("sample dataset")
  st.markdown("Data source: https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf")
  col1, col2, col3 = st.columns(3)
  with col1:
    st.image("images0.JPG", width=100)
  with col2:
    st.image("images1.JPG", width=100)
  with col3:
    st.image("images2.JPG", width=100)

  col1, col2, col3 = st.columns(3)
  with col1:
    st.image("images3.JPG", width=100)
  with col2:
    st.image("images4.JPG", width=100)
  with col3:
    st.image("images5.JPG", width=100)

  col1, col2, col3 = st.columns(3)
  with col1:
    st.image("images6.JPG", width=100)
  with col2:
    st.image("images7.JPG", width=100)
  with col3:
    st.image("images8.JPG", width=100)
