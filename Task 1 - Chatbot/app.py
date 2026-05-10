"""
CodSoft AI Internship — Task 1
Streamlit Web App for NEXA Chatbot

Author: Sreeja Pollreddy | ID: BY25RY287818
Run: streamlit run app.py
"""

import streamlit as st
import time
from chatbot import NexaBot


# ── Page Config ──
st.set_page_config(
    page_title="Nexa — AI Chatbot | CodSoft",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .chat-header {
        text-align: center; padding: 1.5rem 0;
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #a855f7 100%);
        border-radius: 16px; margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.35);
    }
    .chat-header h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .chat-header p { color: rgba(255,255,255,0.9); font-size: 0.95rem; margin: 0.3rem 0 0 0; }
    .status-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255,255,255,0.2); padding: 4px 14px;
        border-radius: 20px; color: white; font-size: 0.85rem; margin-top: 0.5rem;
    }
    .status-dot {
        width: 8px; height: 8px; background: #4ade80; border-radius: 50%;
        display: inline-block; animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
    .sidebar-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0e7ff 100%);
        border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
    }
    .sidebar-card h3 { margin: 0 0 0.5rem 0; font-size: 1rem; color: #333; }
    .sidebar-card p { margin: 0.3rem 0; font-size: 0.85rem; color: #555; }
    .mood-display {
        text-align: center; padding: 0.8rem;
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border-radius: 10px; margin: 0.5rem 0;
    }
    .footer { text-align: center; padding: 1rem; color: #999; font-size: 0.8rem; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ── Session State ──
if "chatbot" not in st.session_state:
    st.session_state.chatbot = NexaBot()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": st.session_state.chatbot.get_greeting()}]
if "message_count" not in st.session_state:
    st.session_state.message_count = 0
if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False


# ── Header ──
st.markdown("""
<div class="chat-header">
    <h1>🤖 Nexa</h1>
    <p>Rule-Based AI Chatbot with Sentiment Awareness</p>
    <div class="status-badge"><span class="status-dot"></span> Online & Ready</div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🤖 About Nexa")
    st.markdown("""
    <div class="sidebar-card">
        <h3>📋 Project Info</h3>
        <p>🏢 <strong>CodSoft</strong> AI Internship</p>
        <p>📝 Task 1: Rule-Based Chatbot</p>
        <p>👩‍💻 By: Sreeja Pollreddy</p>
        <p>🆔 ID: BY25RY287818</p>
    </div>
    """, unsafe_allow_html=True)

    # Mood indicator
    mood_emoji = st.session_state.chatbot.get_mood_emoji()
    st.markdown(f"""
    <div class="mood-display">
        <strong>Current Mood:</strong> {mood_emoji}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Quick Prompts")
    quick_prompts = [
        "👋 Hello!", "😂 Tell me a joke", "🧠 Share a fact",
        "🧮 What is 42 * 58?", "📅 What's the time?", "🤖 What is AI?",
        "💡 Motivate me", "❓ help", "📊 my stats", "🌍 Meaning of life?"
    ]
    for prompt in quick_prompts:
        if st.button(prompt, key=f"btn_{prompt}", use_container_width=True):
            clean = prompt.split(" ", 1)[1] if " " in prompt else prompt
            st.session_state.messages.append({"role": "user", "content": clean})
            response = st.session_state.chatbot.get_response(clean)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.message_count += 1
            st.rerun()

    st.divider()

    # Stats
    st.markdown("### 📊 Live Stats")
    col1, col2 = st.columns(2)
    col1.metric("Messages", st.session_state.message_count)
    topics = len(st.session_state.chatbot.topics_discussed)
    col2.metric("Topics", topics)

    if st.session_state.chatbot.user_name:
        st.info(f"👤 Chatting as: **{st.session_state.chatbot.user_name}**")

    st.divider()

    # Exit & Restart
    if st.button("🚪 Exit Chat", use_container_width=True, type="primary"):
        st.session_state.messages.append({"role": "user", "content": "bye"})
        st.session_state.messages.append({"role": "assistant", "content": "Goodbye! 👋 It was great chatting with you. Have a wonderful day!"})
        st.session_state.conversation_ended = True
        st.rerun()

    if st.button("🗑️ Restart Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = [{"role": "assistant", "content": st.session_state.chatbot.get_greeting()}]
        st.session_state.message_count = 0
        st.session_state.conversation_ended = False
        st.session_state.chatbot = NexaBot()
        st.rerun()

    st.markdown('<div class="footer">Made with ❤️ by Sreeja Pollreddy<br>CodSoft AI Internship © 2026</div>', unsafe_allow_html=True)


# ── Chat Messages ──
for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# ── Chat Input ──
if st.session_state.conversation_ended:
    st.info("💬 Conversation ended. Click **🗑️ Restart Chat** in the sidebar to start a new chat!")
else:
    if prompt := st.chat_input("Type your message here... 💬"):
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = st.session_state.chatbot.get_response(prompt)

        if prompt.lower().strip() in ['bye', 'goodbye', 'exit', 'quit', 'stop', 'end']:
            st.session_state.conversation_ended = True

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            displayed = ""
            for char in response:
                displayed += char
                placeholder.markdown(displayed + "▌")
                time.sleep(0.008)
            placeholder.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.message_count += 1

        if st.session_state.conversation_ended:
            st.rerun()
