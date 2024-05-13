import streamlit as st
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import cv2
import io
import sys
import torch
sys.path.insert(0, '/home/rol3ert99/Pulpit/super-resolution/image_colorizing')
import color_img

SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"


def read_image_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def generate_enhanced_image(upload):
    model = hub.load(SAVED_MODEL_PATH)
    image = read_image_bytes(upload)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image_resolution = image.shape[:2]
    image = np.expand_dims(image, axis=0)
    
    hr_image = model(image)
    hr_image = np.array(hr_image)
    hr_image = hr_image[0,:,:,:]
    hr_image = np.clip(hr_image, 0, 255)
    hr_image = hr_image.astype(np.uint8)
    hr_image_resolution = hr_image.shape[:2]

    return hr_image, image_resolution, hr_image_resolution



st.set_page_config(page_title="Super-Resolution App", layout='wide')

st.title('Welcome to our Super-Resolution App!')

st.markdown("""Our project is a web application that allows for enhancing the 
            resolution of images online. Utilizing advanced artificial 
            intelligence technologies, our tool can significantly increase 
            the quality of images, restoring their detail and sharpness.  
            Discover the potential of artificial intelligence and make your 
            images look better than ever before!  
            [Github](https://github.com/Rol3ert99/super-resolution)  
            * **Authors:** Robert Walery, Konrad Maciejczyk  
            * **Python libraries:** tensorflow, opencv, streamlit, numpy
            """)

option = st.selectbox(
    'what operation you want to perform on the image?',
    ('Super-resolution', 'Colorizing'))


st.write("## Upload file")
my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)

if my_upload is not None and option == 'Super-resolution':
    col1.header("Before")
    col2.header("After")
    col1.image(my_upload, use_column_width=True)

    hr_img, img_resolution, hr_img_resolution = generate_enhanced_image(my_upload)

    col2.image(hr_img, use_column_width=True)

    col1.write("Original resolution: " + str(img_resolution[0]) + 'x' 
               + str(img_resolution[1]))

    col2.write("Received resolution: " + str(hr_img_resolution[0]) + 'x' 
               + str(hr_img_resolution[1]))

    _, buffer = cv2.imencode('.png', cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR))
    img_io = io.BytesIO(buffer)
    col2.download_button(label='Click here to download',
                         data=img_io,
                         file_name='enhanced_image.png',
                         mime='image/png')
    
    
if my_upload is not None and option == 'Colorizing':
    col1.header("Before")
    col2.header("After")
    col1.image(my_upload, use_column_width=True)
    img = read_image_bytes(my_upload)
    colorizer_siggraph17 = color_img.siggraph17(pretrained=True).eval()
    (tens_l_orig, tens_l_rs) = color_img.preprocess_img(img, HW=(256,256))
    img_bw = color_img.postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    out_img_siggraph17 = color_img.postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())    
    col2.image(out_img_siggraph17, use_column_width=True)
