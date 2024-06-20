import streamlit as st
from utils import read_image, predict

st.title('Object Detection App - Slash Intern')

# file uploader
uploaded_file = st.file_uploader("Choose a file", type=['.png', '.jpeg'])
if uploaded_file is not None:
    img = read_image(uploaded_file)
    
if st.button('Analyse Image'):
    try:
        preds = predict(img)
        
        if len(preds) == 0:
            st.header('No objects detected.')
        else:
            st.header('List of detected objects:')
            for pred in preds:
                st.markdown("- " + f'{pred}')
    except:
        st.error("Something went wrong, Please try again", icon="🚨")