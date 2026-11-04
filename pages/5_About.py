import streamlit as st
from utils import setup, hero

setup("About", "ℹ️")
hero("About & Safety", "Dataset, method, limitations and responsible use.")

a, b = st.columns(2)
a.markdown("""<div class="card"><h3>Dataset</h3>
<p>ISIC (International Skin Imaging Collaboration) images from the Kaggle dataset
<i>Skin Cancer Detection using CNN</i>: 2,357 images over 9 classes, with melanoma and nevus
slightly more frequent.</p></div>""", unsafe_allow_html=True)
b.markdown("""<div class="card"><h3>Tech stack</h3>
<p>TensorFlow/Keras CNN trained on Kaggle, Streamlit front-end deployed from GitHub, and an
Anthropic Claude-powered assistant for disease information.</p></div>""", unsafe_allow_html=True)
st.write("")
st.markdown("""<div class="card"><h3>Limitations</h3><ul>
<li>Small dataset; the model may not generalise to different cameras, skin tones or lighting.</li>
<li>Look-alike classes (e.g. melanoma vs nevus) are easily confused.</li>
<li>It outputs probabilities, not diagnoses. A dermatologist (and often a biopsy) is required.</li>
<li>The AI assistant gives general education and can make mistakes.</li>
</ul></div>""", unsafe_allow_html=True)
st.write("")
st.markdown('<div class="disclaimer">If a mole is changing, bleeding, itching or looks different from others, '
            'see a doctor. Do not wait for an app.</div>', unsafe_allow_html=True)
