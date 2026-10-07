import os
import streamlit as st
from openai import OpenAI

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* hide the default Streamlit hamburger & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* chat container */
    .stChatMessage { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Groq client (OpenAI-compatible) ──────────────────────────────────────────
API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL   = "openai/gpt-oss-120b"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, friendly, and concise AI assistant. "
                "Answer clearly and accurately."
            ),
        }
    ]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 AI Chatbot")
st.caption(f"Powered by `{MODEL}` via Groq")

st.divider()

# ── Render existing conversation (skip system message) ───────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Type a message…"):
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        stream = client.chat.completions.create(
            model=MODEL,
            messages=st.session_state.messages,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_response += delta
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    # Persist assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**Model:** `{MODEL}`")
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]  # keep system msg
        st.rerun()
    st.divider()
    st.caption("Built with Streamlit + Groq")
