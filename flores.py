# streamlit_audio_recorder y whisper by Alfredo Diaz - version Mayo 2024

# En VsC seleccione la version de Python (recomiendo 3.9) 
#CTRL SHIFT P  para crear el enviroment (Escriba Python Create Enviroment) y luego venv 

#o puede usar el siguiente comando en el shell
#Vaya a "view" en el menú y luego a terminal y lance un terminal.
#python -m venv env

#Verifique que el terminal inicio con el enviroment o en la carpeta del proyecto active el env.
#cd D:\flores\env\Scripts\
#.\activate 

#Debe quedar asi: (.venv) D:\proyectos_ia\Flores>

#Puedes verificar que no tenga ningun libreria preinstalada con
#pip freeze
#Actualicie pip con pip install --upgrade pip

#pip install tensorflow==2.15 La que tiene instalada Google Colab o con la versión qu fué entrenado el modelo
#Verifique se se instaló numpy, no trate de instalar numpy con pip install numpy, que puede instalar una version diferente
#pip install streamlit
#Verifique se se instaló no trante de instalar con pip install pillow
#Esta instalacion se hace si la requiere pip install opencv-python

#Descargue una foto de una flor que le sirva de ícono 

# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st  
import tensorflow as tf # TensorFlow is required for Keras to work
from PIL import Image
import numpy as np


# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Reconocimiento de Flores",
    page_icon = ":smile:",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) # Oculta el código CSS de la pantalla, ya que están incrustados en el texto de rebajas. Además, permita que Streamlit se procese de forma insegura como HTML

#st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache_resource
def load_model():
    model=tf.keras.models.load_model('./flores_model.h5')
    return model
with st.spinner('Modelo está cargando..'):
    model=load_model()
    


with st.sidebar:
        st.image('rosa.jpg')
        st.title("Reconocimiento de imagen")
        st.header("By: Sofia Higuera")
        st.subheader("Reconocimiento de imagen para flores")
        st.text("Ejercicio de clase")
        st.write("UNAB")
        st.markdown(
             """
             <span style='color:red'> This is First page
             You can:
             - Send email
             - Contact ahiguera334@unab.edu.co </span>
             
             """,
             unsafe_allow_html=True
             )
        confianza=st.slider("Seleccione el nivel de confianza%", 0,100,50)/100

#crear las columnas que centran el logo 
st.image('logo.png')
st.title("Smart Regions Center")

st.write("Somos un equipo apasionado de profesionales dedicados a hacer la diferencia")
st.write("""
         # Detección de flores
         """
         )

#PROGRAMA DE PREDICCIONES:
#CAPTURA DE LA IMAGEN, NO PREPOCESAMIENTO (INICIO DEL PROGRAMA)
#SE LLAMA UNA FUNCIÓN QUE HACE EL RECONOCIMIENTO DE LA IMAGEN O LA PREDICCIÓN, SE DEVUELVE E IMPRIME LOS RESULTADOS


#LA FUNCION ES SOLO PREDICCIÓN
def import_and_predict(image_data, model, class_names):
    
    image_data = image_data.resize((180, 180)) #RESIZE
    
    image = tf.keras.utils.img_to_array(image_data)  #LO CONVIERTO EN UN ARRAY PARA METERLE OTRA DIMENSION
    image = tf.expand_dims(image, 0) # Create a batch  UN DIM ADICIONAL

    
    # Predecir con el modelo
    prediction = model.predict(image) #LA IMAGEN YA ES UN ARREGLO (180,180,3)
    index = np.argmax(prediction)  #DEVUELVAME EL INDICE DEL MAYOR DE TODA LA LISTA
    score = tf.nn.softmax(prediction[0])  #SACAR EL VALOR DEL SCORE
    class_name = class_names[index]  #DEVUELVAME EL NOMBRE DE LA CLASE QUE ESTA EN ESE INDEX
    
    return class_name, score


class_names = open("./clases.txt", "r").readlines()

img_file_buffer = st.camera_input("Capture una foto para identificar una flor")    
if img_file_buffer is None:
    st.text("Por favor tome una foto")
else:
    image = Image.open(img_file_buffer)
    st.image(image, use_column_width=True)
    
    # Realizar la predicción
    class_name, score = import_and_predict(image, model, class_names)
    
    # Mostrar el resultado

    if np.max(score)>0.5:  #>confianza: 
        st.subheader(f"Tipo de Flor: {class_name}")
        st.text(f"Puntuación de confianza: {100 * np.max(score):.2f}%")
    else:
        st.text(f"No se pudo determinar el tipo de flor")