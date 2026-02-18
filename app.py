import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input
from PIL import Image

st.set_page_config(page_title="Dog Breed Identification", page_icon="🐶")

st.title("🐶 Dog Breed Identification using Transfer Learning")
st.write("Upload a dog image and get Top-3 predictions.")

@st.cache_resource
def load_my_model():
    return load_model("dogbreed.h5")

model = load_my_model()

# IMPORTANT: Keep this class order SAME as training
class_names = [
    'affenpinscher','beagle','appenzeller','basset','bluetick','boxer',
    'cairn','doberman','german_shepherd','golden_retriever','kelpie',
    'komondor','leonberg','mexican_hairless','pug','redbone',
    'shih-tzu','toy_poodle','vizsla','whippet'
]

uploaded_file = st.file_uploader("Upload a dog image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((128, 128))
    x = np.expand_dims(np.array(img), axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)[0]
    top3 = np.argsort(preds)[-3:][::-1]

    st.subheader("Top-3 Predictions")
    for i in top3:
        st.write(f"{class_names[i]} : {preds[i]*100:.2f}%")
