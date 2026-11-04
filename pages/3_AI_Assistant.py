import streamlit as st
from utils import setup, hero, DISEASES

setup("AI Assistant", "🤖")
hero("AI Skin Health Assistant", "Ask about the lesion types, symptoms, risk factors and prevention.")

MODEL = "claude-sonnet-5-5"
KB = "\n".join(f"- {n.title()} ({d['risk']}): {d['summary']} Looks: {d['look']} Causes: {d['causes']} Next: {d['next']}"
               for n, d in DISEASES.items())
SYSTEM = f"""You are DermaScan's educational assistant about skin lesions and skin cancer.
Cover these 9 conditions: actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus,
pigmented benign keratosis, seborrheic keratosis, squamous cell carcinoma, vascular lesion.
Rules: give clear, friendly, accurate general education; never diagnose or tell someone a lesion is
safe; for anything suspicious, changing, bleeding or worrying, advise seeing a dermatologist promptly;
politely decline unrelated topics. Keep answers concise. Reply in the user's language
(English or Roman Urdu/Hindi is fine).
Reference notes:
{KB}"""

def get_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return st.session_state.get("user_key", "")

with st.sidebar:
    if not get_key():
        st.text_input("Anthropic API key (optional)", type="password", key="user_key",
                      help="Without a key the assistant uses the built-in knowledge base.")
    if st.button("Clear chat"):
        st.session_state.pop("chat", None)
        st.rerun()

def fallback(q):
    q = q.lower()
    hits = [(n, d) for n, d in DISEASES.items() if n.split()[0] in q or n in q]
    if not hits:
        return ("I can answer questions about these conditions: " +
                ", ".join(n.title() for n in DISEASES) +
                ". Try asking e.g. *What is melanoma?* (Add an API key for open-ended answers.)")
    out = []
    for n, d in hits:
        out.append(f"**{n.title()}** ({d['risk']})\n\n{d['summary']}\n\n- **Looks like:** {d['look']}\n"
                   f"- **Causes/risks:** {d['causes']}\n- **What to do:** {d['next']}")
    return "\n\n---\n\n".join(out)

def stream_reply(msgs):
    import anthropic
    client = anthropic.Anthropic(api_key=get_key())
    with client.messages.stream(model=MODEL, max_tokens=800, system=SYSTEM, messages=msgs) as s:
        for t in s.text_stream:
            yield t

if "chat" not in st.session_state:
    st.session_state.chat = []
    pred = st.session_state.get("last_prediction")
    if pred:
        st.info(f"Your last classifier result: **{pred['label'].title()}** ({pred['confidence']:.0%}). "
                "Ask me about it below.")

for m in st.session_state.chat:
    st.chat_message(m["role"]).write(m["content"])

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
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        if get_key():
            ctx = ""
            pred = st.session_state.get("last_prediction")
            msgs = [dict(m) for m in st.session_state.chat]
            if pred:
                msgs[0]["content"] = (f"[Context: the CNN classified the user's uploaded image as "
                                      f"{pred['label']} ({pred['confidence']:.0%}); this is not a diagnosis.]\n"
                                      + msgs[0]["content"])
            try:
                reply = st.write_stream(stream_reply(msgs))
            except Exception as e:
                reply = fallback(prompt)
                st.warning(f"AI service unavailable ({type(e).__name__}); showing built-in info.")
                st.write(reply)
        else:
            reply = fallback(prompt)
            st.write(reply)
    st.session_state.chat.append({"role": "assistant", "content": reply})
