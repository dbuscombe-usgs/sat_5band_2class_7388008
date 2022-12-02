# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2022, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import streamlit as st
st.set_page_config(
     page_title="satellite 5band 2class model 7388008",
     page_icon="🖖",
     layout="centered",
     initial_sidebar_state="collapsed",
     menu_items={
         'Get Help': None,
         'Report a bug': None,
         'About': "Watermask your Sentinel-2 and Landsat 7/8/9 satellite images!"
     }
 )

import numpy as np
import os,io
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imsave
from skimage.filters import threshold_otsu
# from doodleverse_utils.prediction_imports import *
from doodleverse_utils.imports import standardize, label_to_colors

from glob import glob
import zipfile
from stqdm import stqdm
from PIL import Image
#============================================================
# =========================================================

#load model
filepath = './saved_model'
model = tf.keras.models.load_model(filepath, compile = True)


#segmentation
def do_compute(images_list, use_tta=True, use_otsu=True, dims=(512, 512)):
    N = 2

    for counter, k in enumerate(stqdm(range(len(images_list)))):
        # st.session_state.img_idx=k

        input_img = images_list[k]
        print(input_img)

        if use_otsu:
            print("Use Otsu threshold")
        else:
            print("No Otsu threshold")

        if use_tta:
            print("Use TTA")
        else:
            print("Do not use TTA")

        # image, worig, horig, input_img = seg_file2tensor_ND(input_img, dims)

        with np.load(input_img) as data:
            input_img = data["arr_0"].astype("uint8")

        worig, horig, channels = input_img.shape

        w, h = dims[0], dims[1]

        print("Original dimensions {}x{}".format(worig,horig))
        print("New dimensions {}x{}".format(w,h))

        img = standardize(input_img)
        
        img = resize(img, dims, preserve_range=True, clip=True) 
        
        img = np.expand_dims(img,axis=0)
        
        est_label = model.predict(img)

        if use_tta:
            #Test Time Augmentation
            est_label2 = np.flipud(model.predict((np.flipud(img)), batch_size=1))
            est_label3 = np.fliplr(model.predict((np.fliplr(img)), batch_size=1))
            est_label4 = np.flipud(np.fliplr(model.predict((np.flipud(np.fliplr(img))))))

            #soft voting - sum the softmax scores to return the new TTA estimated softmax scores
            est_label = est_label + est_label2 + est_label3 + est_label4
            est_label /= 4
        
        pred = np.squeeze(est_label, axis=0)
        pred = resize(pred, (worig, horig), preserve_range=True, clip=True)
        
        if use_otsu:
            water = pred[:,:,0]
            thres = threshold_otsu(water)
            print("Otsu threshold is {}".format(thres))
            mask = (water>thres).astype('uint8')
        else:
            mask = np.argmax(pred,-1)

        # input_img  = np.array(Image.open(input_img), dtype=np.uint8)

        # w = input_img.shape[0]
        # h = input_img.shape[1]       
        # img = standardize(input_img)
        
        # img = resize(img, dims, preserve_range=True, clip=True) 
        
        # img = np.expand_dims(img,axis=0)
        
        # est_label = model.predict(img)

        # #Test Time Augmentation
        # est_label2 = np.flipud(model.predict((np.flipud(img)), batch_size=1))
        # est_label3 = np.fliplr(model.predict((np.fliplr(img)), batch_size=1))
        # est_label4 = np.flipud(np.fliplr(model.predict((np.flipud(np.fliplr(img))))))

        # #soft voting - sum the softmax scores to return the new TTA estimated softmax scores
        # est_label = est_label + est_label2 + est_label3 + est_label4
        # est_label /= 4
        
        # pred = np.squeeze(est_label, axis=0)
        # pred = resize(pred, (w, h), preserve_range=True, clip=True)
        
        # bias=.1
        # thres_land = threshold_otsu(pred[:,:,1])-bias
        # print("Land threshold: %f" % (thres_land))
        # mask = (pred[:,:,1]>=thres_land).astype('uint8')

        root = input_img.split(".")[0]
        imsave(root+"greyscale_out_"+str(counter)+".png", mask*255)
        
        class_label_colormap = [
            "#3366CC",
            "#DC3912",
            "#FF9900",
            "#109618",
            "#990099",
            "#0099C6",
            "#DD4477",
            "#66AA00",
            "#B82E2E",
            "#316395",
        ]
        
        # add classes
        class_label_colormap = class_label_colormap[:2]

        color_label = label_to_colors(
            mask,
            input_img[:, :, 0] == 0,
            alpha=128,
            colormap=class_label_colormap,
            color_class_offset=0,
            do_alpha=False,
        )
        
        imsave(root+"color_out_"+str(counter)+".png", color_label)
        
        #overlay plot
        plt.clf()
        plt.imshow(input_img,cmap='gray')
        plt.imshow(color_label, alpha=0.4)
        plt.axis("off")
        plt.margins(x=0, y=0)
        plt.savefig(root+"overlay_out_"+str(counter)+".png", dpi=300, bbox_inches="tight")    


# =========================================================
def rm_thumbnails():
    try:
        for k in glob('*_out*'):
            os.remove(k)
    except:
        pass

def create_zip():
    with zipfile.ZipFile('results.zip', mode="w") as archive:
        for k in glob("*_out*"):
            archive.write(k)
    
    with open('results.zip','rb') as f:
        g=io.BytesIO(f.read()) 
    os.remove('results.zip')
    rm_thumbnails()
    return g

def compute_button():
    do_compute(images_list)
    st.balloons()

# =========================================================
# ================draw page ==============


st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload files. Works with 5-band npz format files only", accept_multiple_files=True)
for uploaded_file in uploaded_files:
     bytes_data = uploaded_file.read()
images_list=uploaded_files

# Initialize app states
if 'img_idx' not in st.session_state:
    st.session_state.img_idx=0


st.title("satellite 5band 2class model 7388008")
st.markdown("by [Daniel Buscombe](https://github.com/dbuscombe-usgs), [Marda Science](https://www.mardascience.com/) ")


col1,col2,col3,col4=st.columns(4)
with col1:
    st.button(label="Compute",key="compute_button",on_click=compute_button)

with col4:
    st.download_button(
     label="Download zipped folder of results",
     data=create_zip(),
     file_name= 'results.zip', 
 )

#

















