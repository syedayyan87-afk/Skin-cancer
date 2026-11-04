import json, os
import pandas as pd
import streamlit as st
from utils import setup, hero

setup("Model Performance", "📊")
hero("Model Performance", "Training curves, confusion matrix and per-class metrics from the Kaggle run.")

A = "assets"

def show_image(name):
    p = f"{A}/{name}"
    if os.path.exists(p):
        st.image(p, use_container_width=True)
    else:
        st.info(f"Add {p} from the notebook output.")

tab1, tab2, tab3 = st.tabs(["Loss / Accuracy", "Confusion matrix", "Classification report"])
with tab1:
    show_image("loss_curve.png")
with tab2:
    show_image("confusion_matrix.png")
with tab3:
    p = f"{A}/metrics.json"
    if os.path.exists(p):
        with open(p) as f:
            rep = json.load(f)
        acc = rep.get("accuracy")
        if acc is not None:
            st.metric("Overall test accuracy", f"{acc:.1%}")
        df = pd.DataFrame(rep).T.drop(index="accuracy", errors="ignore").round(3)
        st.dataframe(df, use_container_width=True)
    else:
        st.info(f"Add {p} from the notebook output.")

st.subheader("Architecture")
st.code("""Input 180x180x3 -> Augmentation -> Rescaling
4 x [Conv2D(3x3, ReLU) -> BatchNorm -> MaxPool]  (32, 64, 128, 256 filters)
GlobalAveragePooling -> Dense(256) -> Dropout(0.5) -> Dense(9, softmax)
Adam 1e-3, EarlyStopping, ReduceLROnPlateau""", language="text")
