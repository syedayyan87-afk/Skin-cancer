import json
import numpy as np
import streamlit as st
from PIL import Image

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {
  padding: 2.6rem 2.2rem; border-radius: 24px; margin-bottom: 1.5rem;
  background: radial-gradient(circle at 15% 20%, rgba(124,92,255,.35), transparent 45%),
              radial-gradient(circle at 85% 80%, rgba(0,212,255,.25), transparent 45%),
              #131A2B;
  border: 1px solid rgba(255,255,255,.08);
}
.hero h1 {font-size: 2.6rem; margin: 0 0 .4rem 0;
  background: linear-gradient(90deg,#B9A8FF,#5CE1FF); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;}
.hero p {color: #AEB8D0; font-size: 1.08rem; margin: 0;}
.card {
  background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
  border-radius: 18px; padding: 1.2rem 1.3rem; height: 100%;
  transition: transform .2s, border-color .2s;
}
.card:hover {transform: translateY(-3px); border-color: rgba(124,92,255,.7);}
.card h3 {margin: 0 0 .4rem 0; font-size: 1.1rem;}
.card p, .card li {color: #AEB8D0; font-size: .95rem;}
.stat {text-align: center;}
.stat .num {font-size: 2.2rem; font-weight: 800;
  background: linear-gradient(90deg,#B9A8FF,#5CE1FF); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;}
.stat .lbl {color: #AEB8D0; font-size: .85rem; letter-spacing: .05em; text-transform: uppercase;}
.badge {display: inline-block; padding: .15rem .7rem; border-radius: 999px;
  font-size: .75rem; font-weight: 700; letter-spacing: .04em;}
.b-mal {background: rgba(255,82,82,.18); color: #FF8A8A; border: 1px solid #FF5252;}
.b-pre {background: rgba(255,171,64,.18); color: #FFC37A; border: 1px solid #FFAB40;}
.b-ben {background: rgba(76,217,100,.16); color: #7DE795; border: 1px solid #4CD964;}
.disclaimer {background: rgba(255,171,64,.10); border-left: 4px solid #FFAB40;
  padding: .8rem 1rem; border-radius: 8px; color: #FFD9A0; font-size: .9rem;}
</style>
"""

def setup(title, icon):
    st.set_page_config(page_title=f"{title} | DermaScan AI", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### 🧬 DermaScan AI")
        st.caption("CNN skin-lesion classifier + AI assistant")
        st.markdown('<div class="disclaimer">Educational project only. Not a medical '
                    'diagnosis. See a dermatologist for any skin concern.</div>',
                    unsafe_allow_html=True)

def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)

def badge(risk):
    cls = {"Malignant": "b-mal", "Precancerous": "b-pre", "Benign": "b-ben"}[risk]
    return f'<span class="badge {cls}">{risk.upper()}</span>'

# General educational info (not medical advice)
DISEASES = {
    "actinic keratosis": {
        "risk": "Precancerous",
        "summary": "Rough, scaly patches caused by long-term sun exposure. Considered precancerous because some can progress to squamous cell carcinoma.",
        "look": "Dry, scaly or crusty patch, often pink, red or brown; commonly on face, scalp, ears, hands.",
        "causes": "Cumulative UV exposure, fair skin, age, weakened immune system.",
        "next": "Dermatologist review. Treatments include cryotherapy, topical creams or procedures. Daily sunscreen helps prevention."},
    "basal cell carcinoma": {
        "risk": "Malignant",
        "summary": "The most common skin cancer. It grows slowly and rarely spreads, but can damage nearby tissue if untreated.",
        "look": "Pearly or waxy bump, flat flesh-coloured scar-like patch, or a sore that bleeds and does not heal.",
        "causes": "UV exposure (sun and tanning beds), fair skin, prior sunburns.",
        "next": "Needs prompt medical evaluation. Usually treated by surgical removal, Mohs surgery or other local treatment."},
    "dermatofibroma": {
        "risk": "Benign",
        "summary": "A common, harmless firm skin nodule, often on the legs.",
        "look": "Small firm brown/pink bump that dimples inward when pinched.",
        "causes": "Often follows a minor injury such as an insect bite; exact cause unclear.",
        "next": "Usually needs no treatment. See a doctor if it changes, grows quickly or becomes painful."},
    "melanoma": {
        "risk": "Malignant",
        "summary": "The most dangerous skin cancer because it can spread to other organs. Early detection greatly improves outcomes.",
        "look": "Use ABCDE: Asymmetry, irregular Border, uneven Color, Diameter over ~6 mm, and Evolving shape/size/colour.",
        "causes": "UV exposure, many moles, family history, fair skin, history of blistering sunburns.",
        "next": "Any suspicious or changing mole should be seen by a dermatologist promptly. Treatment is usually surgical removal, with further therapy depending on stage."},
    "nevus": {
        "risk": "Benign",
        "summary": "A common mole made of pigment cells. Most are harmless, but some can change over time, so monitoring matters.",
        "look": "Round or oval, even colour, smooth edges, usually small and stable.",
        "causes": "Genetics and sun exposure.",
        "next": "Monitor with regular self-checks. Get any mole that changes checked by a doctor."},
    "pigmented benign keratosis": {
        "risk": "Benign",
        "summary": "A non-cancerous pigmented growth (often related to seborrheic keratosis or solar lentigo).",
        "look": "Brown or tan, slightly raised or flat, sharply defined patch.",
        "causes": "Ageing and sun exposure.",
        "next": "Harmless; can be removed for cosmetic reasons. Have a doctor check it if unsure, as it can resemble melanoma."},
    "seborrheic keratosis": {
        "risk": "Benign",
        "summary": "One of the most common benign skin growths in adults, often appearing with age.",
        "look": "Waxy, 'stuck-on' brown, black or tan growth with a slightly rough surface.",
        "causes": "Ageing; tends to run in families. Not caused by a virus and not contagious.",
        "next": "Usually no treatment needed. Removal is possible if irritated or for cosmetic reasons."},
    "squamous cell carcinoma": {
        "risk": "Malignant",
        "summary": "The second most common skin cancer. Can invade deeper tissue and sometimes spread if untreated.",
        "look": "Firm red nodule, scaly crusted patch, or a sore that does not heal, often on sun-exposed areas.",
        "causes": "Long-term UV exposure, actinic keratosis, weakened immune system, fair skin.",
        "next": "Needs prompt medical evaluation; usually treated with surgical removal or other procedures."},
    "vascular lesion": {
        "risk": "Benign",
        "summary": "A group of skin marks made of blood vessels, such as angiomas, cherry angiomas and port-wine stains.",
        "look": "Red, purple or blue spots, bumps or patches that may blanch under pressure.",
        "causes": "Developmental or age-related changes in blood vessels.",
        "next": "Usually harmless; laser treatment is an option. See a doctor if one bleeds repeatedly or changes."},
}

def info_for(name):
    return DISEASES.get(name.lower().replace("_", " ").strip())

@st.cache_resource(show_spinner="Loading CNN model...")
def load_model():
    from tensorflow import keras
    model = keras.models.load_model("skin_cancer_cnn.keras")
    with open("class_names.json") as f:
        names = json.load(f)
    return model, names

def predict(img: Image.Image):
    model, names = load_model()
    size = model.input_shape[1:3]
    x = np.expand_dims(np.asarray(img.convert("RGB").resize(size), dtype="float32"), 0)
    probs = model.predict(x, verbose=0)[0]
    return names, probs
