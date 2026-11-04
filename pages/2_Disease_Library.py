import streamlit as st
from utils import setup, hero, badge, DISEASES

setup("Disease Library", "📚")
hero("Disease Library", "Plain-language information on all 9 lesion types in the dataset.")

q, f = st.columns([2, 1])
query = q.text_input("Search", placeholder="e.g. melanoma, mole, sun...")
risk = f.multiselect("Risk level", ["Malignant", "Precancerous", "Benign"],
                     default=["Malignant", "Precancerous", "Benign"])

shown = 0
for name, d in DISEASES.items():
    text = (name + " " + " ".join(d.values())).lower()
    if d["risk"] not in risk or (query and query.lower() not in text):
        continue
    shown += 1
    with st.expander(f"{name.title()}", expanded=False):
        st.markdown(badge(d["risk"]), unsafe_allow_html=True)
        st.write(d["summary"])
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="card"><h3>What it looks like</h3><p>{d["look"]}</p></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="card"><h3>Common causes / risks</h3><p>{d["causes"]}</p></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="card"><h3>What to do</h3><p>{d["next"]}</p></div>', unsafe_allow_html=True)
if not shown:
    st.info("No matching conditions.")

st.write("")
st.subheader("ABCDE rule for moles")
cols = st.columns(5)
for col, (l, t) in zip(cols, [("A", "Asymmetry"), ("B", "Border irregular"), ("C", "Color varies"),
                              ("D", "Diameter > 6mm"), ("E", "Evolving")]):
    col.markdown(f'<div class="card stat"><div class="num">{l}</div><div class="lbl">{t}</div></div>',
                 unsafe_allow_html=True)
