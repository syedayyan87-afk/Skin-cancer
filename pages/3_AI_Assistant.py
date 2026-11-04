import base64, io, re
import streamlit as st
from PIL import Image
from utils import setup, hero, badge, DISEASES, predict, info_for

setup("AI Assistant", "🤖")
hero("AI Skin Health Assistant", "Ask about lesion types, symptoms, risk factors and prevention, or attach a skin image.")

MODEL = "claude-sonnet-5-5"
KB = "\n".join(f"- {n.title()} ({d['risk']}): {d['summary']} Looks: {d['look']} Causes: {d['causes']} Next: {d['next']}"
               for n, d in DISEASES.items())
SYSTEM = f"""You are DermaScan's educational assistant about skin lesions and skin cancer.
Cover these 9 conditions: actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus,
pigmented benign keratosis, seborrheic keratosis, squamous cell carcinoma, vascular lesion.
Rules: give clear, friendly, accurate general education; never diagnose or tell someone a lesion is
safe; if an image is attached, describe only what is visible (colour, border, symmetry, size cues),
mention the CNN result if provided as an unreliable estimate, and for anything suspicious, changing,
bleeding or worrying advise seeing a dermatologist promptly. Politely decline unrelated topics.
Keep answers concise. Reply in the user's language (English or Roman Urdu/Hindi is fine).
Reference notes:
{KB}"""

# ---------- helpers ----------
def get_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return st.session_state.get("user_key", "")

def prep(file):
    im = Image.open(file).convert("RGB")
    im.thumbnail((768, 768))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return im, buf.getvalue()

def cnn_top3(im):
    try:
        names, probs = predict(im)
    except Exception:
        return None
    order = probs.argsort()[::-1][:3]
    return [(names[i], float(probs[i])) for i in order]

ALIASES = {
    "actinic keratosis": ["actinic", "solar keratosis", "ak"],
    "basal cell carcinoma": ["basal", "bcc"],
    "dermatofibroma": ["dermatofibroma"],
    "melanoma": ["melanoma"],
    "nevus": ["nevus", "naevus", "mole", "moles"],
    "pigmented benign keratosis": ["pigmented"],
    "seborrheic keratosis": ["seborrheic", "seborrhoeic", "sk"],
    "squamous cell carcinoma": ["squamous", "scc"],
    "vascular lesion": ["vascular", "angioma", "hemangioma", "port wine", "port-wine"],
}
TOPICS = [
    (["prevent", "prevention", "protect", "sunscreen", "sun", "bachao", "bachna", "dhoop"],
     "**Prevention**\n\n- Use broad-spectrum sunscreen (SPF 30+) daily and reapply every ~2 hours outdoors.\n"
     "- Seek shade at midday, wear a hat, sunglasses and covering clothes.\n- Avoid tanning beds.\n"
     "- Do a monthly self-check of your skin and get regular checks if you are high risk."),
    (["abcde", "self-check", "self check", "examine", "check mole"],
     "**ABCDE rule for moles**\n\n- **A**symmetry, **B**order irregular, **C**olor uneven, "
     "**D**iameter over ~6 mm, **E**volving (changing).\n\nAny mole that fits these or changes should be seen by a dermatologist."),
    (["doctor", "dermatologist", "biopsy", "kab", "when should"],
     "**When to see a doctor**\n\nSee a dermatologist promptly for a lesion that is new, changing in size/shape/colour, "
     "bleeding, itching, crusting, or a sore that does not heal in a few weeks. Only a clinician (often with a biopsy) can diagnose."),
    (["malignant", "benign", "dangerous", "khatarnak", "cancer", "serious"],
     "**Malignant vs benign**\n\n- Malignant here: melanoma, basal cell carcinoma, squamous cell carcinoma.\n"
     "- Precancerous: actinic keratosis.\n- Benign: nevus, dermatofibroma, seborrheic keratosis, pigmented benign keratosis, vascular lesion.\n\n"
     "Melanoma is the most dangerous because it can spread, so early detection matters."),
    (["treat", "treatment", "ilaj", "cure", "surgery", "remove", "removal"],
     "**Treatment (general)**\n\nDepends on the type: skin cancers are usually removed surgically (or with Mohs surgery); "
     "actinic keratosis may be frozen or treated with creams; many benign lesions need no treatment. A dermatologist decides."),
]

def has(q, words):
    return any(re.search(r"\b" + re.escape(w) + r"\b", q) for w in words)

def fallback(q):
    q = q.lower()
    hits = [n for n, al in ALIASES.items() if has(q, al)]
    if not hits and "keratosis" in q:
        hits = ["actinic keratosis", "seborrheic keratosis", "pigmented benign keratosis"]
    if hits:
        out = []
        for n in hits:
            d = DISEASES[n]
            out.append(f"**{n.title()}** ({d['risk']})\n\n{d['summary']}\n\n- **Looks like:** {d['look']}\n"
                       f"- **Causes/risks:** {d['causes']}\n- **What to do:** {d['next']}")
        return "\n\n---\n\n".join(out)
    for words, text in TOPICS:
        if has(q, words):
            return text
    return ("I am running in **basic mode** (no AI key set), so I can only answer about specific conditions "
            "(e.g. *What is melanoma?*) and topics like prevention, ABCDE rule, treatment or when to see a doctor.\n\n"
            "For open-ended answers and image analysis, add an Anthropic API key in the sidebar.")

def build_msgs():
    out = []
    pred = st.session_state.get("last_prediction")
    for i, m in enumerate(st.session_state.chat):
        text = m["content"]
        if m.get("cnn"):
            text = f"[Context: CNN top-3 for the attached image: {m['cnn']}. This is an unreliable estimate, not a diagnosis.]\n{text}"
        elif i == 0 and pred and m["role"] == "user":
            text = f"[Context: classifier result earlier: {pred['label']} ({pred['confidence']:.0%}); not a diagnosis.]\n{text}"
        if m.get("image"):
            b64 = base64.b64encode(m["image"]).decode()
            out.append({"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": text}]})
        else:
            out.append({"role": m["role"], "content": text})
    return out

def stream_reply(msgs):
    import anthropic
    client = anthropic.Anthropic(api_key=get_key())
    with client.messages.stream(model=MODEL, max_tokens=900, system=SYSTEM, messages=msgs) as s:
        for t in s.text_stream:
            yield t

def show_msg(m):
    with st.chat_message(m["role"]):
        if m.get("image"):
            st.image(m["image"], width=260)
        st.write(m["content"])

# ---------- sidebar ----------
with st.sidebar:
    if not get_key():
        st.text_input("Anthropic API key", type="password", key="user_key",
                      help="Without a key the assistant runs in basic (offline) mode.")
        st.caption("Mode: basic (offline)")
    else:
        st.caption("Mode: AI (Claude)")
    if st.button("Clear chat"):
        st.session_state.pop("chat", None)
        st.rerun()

# ---------- state ----------
st.session_state.setdefault("chat", [])
st.session_state.setdefault("up_n", 0)

if not st.session_state.chat:
    pred = st.session_state.get("last_prediction")
    if pred:
        st.info(f"Your last classifier result: **{pred['label'].title()}** ({pred['confidence']:.0%}). Ask me about it below.")

with st.expander("📎 Attach a skin image (optional)", expanded=not st.session_state.chat):
    up = st.file_uploader("Image", type=["jpg", "jpeg", "png"], key=f"up_{st.session_state.up_n}",
                          label_visibility="collapsed")
    if up:
        st.image(up, width=220)
        if st.button("Analyse this image"):
            st.session_state["queued"] = "Please analyse this image and explain what the result means."

for m in st.session_state.chat:
    show_msg(m)

if not st.session_state.chat:
    st.write("Try:")
    cols = st.columns(3)
    for col, s in zip(cols, ["What is melanoma?", "How do I tell a mole from melanoma?",
                             "How can I prevent skin cancer?"]):
        if col.button(s, use_container_width=True):
            st.session_state["queued"] = s
            st.rerun()

prompt = st.chat_input("Ask about a skin condition...") or st.session_state.pop("queued", None)

if prompt:
    msg = {"role": "user", "content": prompt}
    top = None
    if up:
        im, jpg = prep(up)
        msg["image"] = jpg
        top = cnn_top3(im)
        if top:
            msg["cnn"] = ", ".join(f"{n} {p:.0%}" for n, p in top)
            st.session_state["last_prediction"] = {"label": top[0][0], "confidence": top[0][1], "top3": top}
        st.session_state.up_n += 1  # resets the uploader
    st.session_state.chat.append(msg)
    show_msg(msg)

    with st.chat_message("assistant"):
        if get_key():
            try:
                reply = st.write_stream(stream_reply(build_msgs()))
            except Exception as e:
                reply = fallback(prompt)
                st.warning(f"AI service error ({type(e).__name__}); showing basic answer.")
                st.write(reply)
        else:
            parts = []
            if up and top:
                n, p = top[0]
                info = info_for(n)
                parts.append(f"**CNN estimate:** {n.title()} ({p:.0%}), then "
                             + ", ".join(f"{a.title()} ({b:.0%})" for a, b in top[1:])
                             + (f"\n\n{info['summary']}" if info else "")
                             + "\n\nThis is an estimate, not a diagnosis. Please see a dermatologist.")
            elif up:
                parts.append("I could not run the CNN on this image (model files missing).")
            parts.append(fallback(prompt))
            reply = "\n\n---\n\n".join(parts)
            st.write(reply)
    st.session_state.chat.append({"role": "assistant", "content": reply})
    if up:
        st.rerun()
