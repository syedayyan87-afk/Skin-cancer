import streamlit as st
from utils import setup, hero, badge, DISEASES

setup("Home", "🧬")
hero("DermaScan AI",
     "Deep-learning skin lesion analysis across 9 ISIC diagnostic classes, "
     "with an AI assistant to explain what you see.")

c = st.columns(4)
for col, (num, lbl) in zip(c, [("9", "Lesion classes"), ("2,357", "ISIC images"),
                               ("CNN", "Custom architecture"), ("AI", "Disease assistant")]):
    col.markdown(f'<div class="card stat"><div class="num">{num}</div>'
                 f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.write("")
st.subheader("How it works")
a, b, d = st.columns(3)
a.markdown('<div class="card"><h3>1 · Upload</h3><p>Add a clear, well-lit photo of a skin lesion in the <b>Classifier</b> page.</p></div>', unsafe_allow_html=True)
b.markdown('<div class="card"><h3>2 · Analyse</h3><p>A convolutional neural network predicts the most likely class with confidence scores.</p></div>', unsafe_allow_html=True)
d.markdown('<div class="card"><h3>3 · Learn</h3><p>Read the <b>Disease Library</b> or ask the <b>AI Assistant</b> follow-up questions.</p></div>', unsafe_allow_html=True)

st.write("")
st.subheader("Classes covered")
cols = st.columns(3)
for i, (name, d_) in enumerate(DISEASES.items()):
    cols[i % 3].markdown(
        f'<div class="card" style="margin-bottom:12px"><h3>{name.title()}</h3>'
        f'{badge(d_["risk"])}</div>', unsafe_allow_html=True)

st.write("")
st.markdown('<div class="disclaimer">This tool is a student/educational project. Predictions can be wrong and '
            'must never replace professional medical advice.</div>', unsafe_allow_html=True)
