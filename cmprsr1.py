import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

from PIL import Image

import streamlit as st
img_array = None

st.header('SVD Image Compressor')

img_file_buffer = st.file_uploader('Upload image to be compressed: ', type=['png','jpg','jpeg'])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

image0 = img_array[:,:,0]
image1 = img_array[:,:,1]
image2 = img_array[:,:,2]

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image(image, width=250, caption='Uploaded Image')

with col3:
    st.write(' ')

rank = st.number_input('Enter the rank approximation: ')
rank = int(rank)
u0,sigma0,vt0 = np.linalg.svd(image0)
u1,sigma1,vt1 = np.linalg.svd(image1)
u2,sigma2,vt2 = np.linalg.svd(image2)

Sigma0=np.zeros((1024,1024))
Sigma1=np.zeros((1024,1024))
Sigma2=np.zeros((1024,1024))


for i in range(rank):
    Sigma0[i][i] = sigma0[i]
    Sigma1[i][i] = sigma1[i]
    Sigma2[i][i] = sigma2[i]

i0_rank = u0@Sigma0@vt0
i1_rank = u1@Sigma1@vt1
i2_rank = u2@Sigma2@vt2

compressed = np.dstack((i0_rank,i1_rank,i2_rank)).astype(np.uint8)

st.markdown(f'### Rank {rank} approximated Image')
fig = plt.figure()
plt.imshow(compressed, cmap='Greys_r')
st.pyplot(fig)
