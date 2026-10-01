import os
import sys
import string
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app.classifier import classifier

# ==============================================================================
# GENERAL CONVERSATION DATA & LOGIC (DIRECTLY DEFINED IN CODE)
# ==============================================================================

GREETING_RESPONSE = "Hello! 👋 How can I help you with your university-related questions today?"
HOW_ARE_YOU_RESPONSE = "I'm doing great, thank you! 😊 How can I assist you with your campus queries today?"
IDENTITY_RESPONSE = "I'm CampusBot, your digital university student support assistant. I can help answer questions about courses, registration, financial aid, campus facilities, and policies!"
THANKS_RESPONSE = "You're very welcome! 😊 Feel free to ask if you have any more university-related questions."
GOODBYE_RESPONSE = "Goodbye! 👋 Have a wonderful day and best of luck with your studies!"
HELP_RESPONSE = "I'm here to assist you! You can ask me about course registration, financial aid, library hours, IT support, campus facilities, exam schedules, and academic advising."
CONFIRMATION_RESPONSE = "Great! Let me know whenever you have any other questions."
POSITIVE_RESPONSE = "Glad to hear that! 😊 Is there anything else about university services I can help you with?"
NEGATIVE_RESPONSE = "No problem! Have a great day ahead, and come back anytime if you need help."
SMALL_TALK_RESPONSE = "Thank you! I'm an AI student support assistant created to help you navigate campus life and academic inquiries easily."

# Categories and manually defined general conversation phrases
GENERAL_PHRASES_BY_CATEGORY = {
    "Greetings": (
        GREETING_RESPONSE,
        [
            "hello", "helo", "hi", "hii", "hiii", "hey", "heyy", "heyyy",
            "hello there", "hi there", "hey there", "good morning", "good afternoon",
            "good evening", "morning", "evening", "greetings", "howdy"
        ]
    ),
    "How are you": (
        HOW_ARE_YOU_RESPONSE,
        [
            "how are you", "how are u", "how r you", "how r u", "hw r u",
            "how are you doing", "how are you doing today", "are you okay",
            "are you doing well", "how is it going", "hows it going", "how are things"
        ]
    ),
    "Identity": (
        IDENTITY_RESPONSE,
        [
            "who are you", "what are you", "who r u", "what is your name",
            "whats your name", "what's your name", "tell me about yourself",
            "are you a chatbot", "are you a bot", "who made you", "what can you do",
            "what do you do", "introduce yourself", "what is campusbot", "what is this bot"
        ]
    ),
    "Thanks": (
        THANKS_RESPONSE,
        [
            "thank you", "thankyou", "thanks", "thank u", "thankyou so much",
            "thanks a lot", "thnx", "tnx", "many thanks", "thats helpful",
            "that's helpful", "that is helpful", "appreciate it", "thank you very much",
            "awesome thank you"
        ]
    ),
    "Goodbye": (
        GOODBYE_RESPONSE,
        [
            "bye", "byee", "byeee", "goodbye", "good bye", "see you", "see ya",
            "see you later", "talk to you later", "have a good day", "have a nice day",
            "bye bye"
        ]
    ),
    "Help / capabilities": (
        HELP_RESPONSE,
        [
            "help", "help me", "can you help me", "i need help", "assistance",
            "what can i ask you", "how to use this", "options", "show menu",
            "what questions can i ask", "give me info", "support"
        ]
    ),
    "Confirmation / acknowledgement": (
        CONFIRMATION_RESPONSE,
        [
            "ok", "okay", "got it", "alright", "understood", "i see", "sure",
            "cool", "makes sense", "k"
        ]
    ),
    "Positive responses": (
        POSITIVE_RESPONSE,
        [
            "great", "awesome", "perfect", "wonderful", "amazing", "good",
            "nice", "super", "excellent", "brilliant"
        ]
    ),
    "Negative responses": (
        NEGATIVE_RESPONSE,
        [
            "no", "nope", "nah", "no thanks", "no thank you", "not now",
            "nothing", "no further questions", "don't need help", "dont need help"
        ]
    ),
    "Small talk": (
        SMALL_TALK_RESPONSE,
        [
            "nice to meet you", "pleasure to meet you", "good job", "well done",
            "you are helpful", "you are smart", "are you human", "tell me a joke",
            "are you real", "who is your creator", "are you busy", "how old are you"
        ]
    )
}

def normalize_text(text: str) -> str:
    """Normalize user input text for exact general conversation matching.
    Converts to lowercase, removes leading/trailing spaces, and strips basic punctuation.
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation)).strip()
    text = " ".join(text.split())
    return text

# Build normalized lookup map: normalized_phrase -> (category, response)
GENERAL_CONVERSATION_MAP = {}

for category_name, (response_text, phrase_list) in GENERAL_PHRASES_BY_CATEGORY.items():
    for phrase in phrase_list:
        norm_key = normalize_text(phrase)
        if norm_key and norm_key not in GENERAL_CONVERSATION_MAP:
            GENERAL_CONVERSATION_MAP[norm_key] = (category_name, response_text)

def check_general_conversation(raw_prompt: str):
    """Check if the user prompt exactly matches a general conversation phrase after normalization.
    Returns a result dict if matched, or None if not matched.
    """
    normalized_prompt = normalize_text(raw_prompt)
    if normalized_prompt in GENERAL_CONVERSATION_MAP:
        category, response_text = GENERAL_CONVERSATION_MAP[normalized_prompt]
        return {
            "question": raw_prompt,
            "intent": f"general_conversation ({category})",
            "confidence": 1.0,
            "matched_question": normalized_prompt,
            "similarity": 1.0,
            "response": response_text
        }
    return None

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="University Student Support Chatbot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_classifier():
    """Pre-load ML resources once on app launch."""
    classifier.load_resources()
    return classifier


# Initialize global classifier instance
bot_classifier = load_classifier()

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

    # Predict response using General Conversation or existing ML classifier
    with st.spinner("Searching for answer..."):
        general_match = check_general_conversation(prompt)
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


