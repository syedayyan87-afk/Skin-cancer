import streamlit as st
from PIL import Image
from utils import setup, hero, badge, predict, info_for

setup("Classifier", "🔬")
hero("Lesion Classifier", "Upload an image and let the CNN estimate the lesion type.")

left, right = st.columns([1, 1.2], gap="large")
with left:
    src = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
    file = (st.file_uploader("Skin lesion image", type=["jpg", "jpeg", "png"])
            if src == "Upload" else st.camera_input("Take a photo"))
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, use_container_width=True)

with right:
    if file:
        with st.spinner("Analysing..."):
            names, probs = predict(img)
        order = probs.argsort()[::-1]
        top = int(order[0])
        info = info_for(names[top])
        st.session_state["last_prediction"] = {
            "label": names[top], "confidence": float(probs[top]),
            "top3": [(names[i], float(probs[i])) for i in order[:3]]}

        st.markdown(f"### {names[top].title()}")
        if info:
            st.markdown(badge(info["risk"]), unsafe_allow_html=True)
        st.metric("Confidence", f"{probs[top]:.1%}")
        if probs[top] < 0.5:
            st.warning("Low confidence. The model is unsure about this image.")

        st.markdown("**Top 3 predictions**")
        for i in order[:3]:
            st.progress(float(probs[i]), text=f"{names[i].title()}: {probs[i]:.1%}")
        with st.expander("All class probabilities"):
            st.bar_chart({names[i].title(): float(probs[i]) for i in order})
        if info:
            st.info(info["summary"])
        st.page_link("pages/3_AI_Assistant.py", label="Ask the AI assistant about this result →", icon="🤖")
    else:
        st.markdown('<div class="card"><h3>Waiting for an image</h3><p>Tips: good lighting, lesion centred and in focus, '
                    'no filters.</p></div>', unsafe_allow_html=True)
