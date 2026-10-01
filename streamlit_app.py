import os
import sys
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app.classifier import classifier
from general_conversation import general_detector

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="University Student Support Chatbot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_all_resources():
    """Pre-load ML resources and General Conversation detector once on app launch."""
    classifier.load_resources()
    # Share SentenceTransformer model instance to optimize RAM & initialization time
    general_detector.load_resources(model=classifier.model)
    return classifier, general_detector


# Initialize global classifier and general conversation instances
bot_classifier, bot_general_detector = load_all_resources()

# Sidebar UI
with st.sidebar:
    st.title("🎓 University Student Support Chatbot")
    st.markdown(
        "Welcome! This AI chatbot assists university students with answers "
        "regarding course registration, campus facilities, academic policies, "
        "financial aid, IT support, exams, and student services."
    )
    st.markdown("---")
    st.info("💡 All answers are dataset-grounded and verified against university support records.")
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface UI
st.title("University Student Support Chatbot")
st.caption("Ask questions about courses, registration, campus facilities, policies, or financial aid.")

# Initialize session state message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your University Student Support Assistant. How can I help you today?"
        }
    ]

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "debug" in message:
            dbg = message["debug"]
            with st.expander("Debug information"):
                st.write(f"**Question:** {dbg.get('question')}")
                st.write(f"**Predicted Intent:** {dbg.get('intent')}")
                st.write(f"**Intent Confidence:** {dbg.get('confidence')}")
                st.write(f"**Matched Question:** {dbg.get('matched_question')}")
                st.write(f"**Question Similarity:** {dbg.get('similarity')}")

# User prompt input handling
if prompt := st.chat_input("Type your question here..."):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Predict response using General Conversation (Exact + Embedding) or existing ML classifier
    with st.spinner("Searching for answer..."):
        general_match = bot_general_detector.predict_general(prompt)
        if general_match:
            result = general_match
        else:
            result = bot_classifier.predict(prompt)
        response_text = result["response"]

    # Append and display assistant response with debug info
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "debug": result
    })
    with st.chat_message("assistant"):
        st.markdown(response_text)
        with st.expander("Debug information"):
            st.write(f"**Question:** {result.get('question')}")
            st.write(f"**Predicted Intent:** {result.get('intent')}")
            st.write(f"**Intent Confidence:** {result.get('confidence')}")
            st.write(f"**Matched Question:** {result.get('matched_question')}")
            st.write(f"**Question Similarity:** {result.get('similarity')}")



