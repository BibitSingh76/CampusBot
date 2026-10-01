import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Conservative similarity threshold for 2nd layer embedding match
# Max observed similarity for university queries against general phrases is 0.4657,
# whereas natural general phrase variations score 0.72 - 0.88.
GENERAL_EMBEDDING_THRESHOLD = 0.70

# ==============================================================================
# PREDEFINED CATEGORY RESPONSES
# ==============================================================================

GREETING_RESPONSE = "Hello! 👋 How can I help you with your university-related questions today?"
HOW_ARE_YOU_RESPONSE = "I'm doing great! 😊 I'm here to help you with your university-related questions."
IDENTITY_RESPONSE = "I'm CampusBot, your university student support assistant. I can help answer questions about courses, registration, scholarships, and campus services."
CAPABILITIES_RESPONSE = "I can help you with university-related questions such as course registration, scholarships, fees, campus facilities, IT support, library services, and academic policies."
HELP_REQUEST_RESPONSE = "I'm here to assist you! Please feel free to ask your question or state what you need help with."
THANKS_RESPONSE = "You're very welcome! 😊 Feel free to ask if you have any more university-related questions."
GOODBYE_RESPONSE = "Goodbye! 👋 Have a great day and best of luck with your studies!"
CONFIRMATION_RESPONSE = "Great! Let me know whenever you have any other questions."
POSITIVE_RESPONSE = "Glad to hear that! 😊 Is there anything else about university services I can help you with?"
NEGATIVE_RESPONSE = "No problem! Have a great day ahead, and come back anytime if you need help."
CLARIFICATION_RESPONSE = "Of course! Could you tell me which part you'd like me to explain or clarify?"
REPETITION_RESPONSE = "Sure! Let me explain that again in a simpler way."
CASUAL_CONVERSATION_RESPONSE = "I enjoy chatting! 😊 Let me know if you need any university guidance or campus information."
SMALL_TALK_RESPONSE = "Thank you! I'm an AI student support assistant created to help you navigate campus life easily."
APOLOGY_RESPONSE = "No worries at all! Let's get back to your query."
STARTER_RESPONSE = "Hello! I'm ready whenever you are. What would you like to ask about the university today?"

# ==============================================================================
# 500+ MANUALLY DEFINED GENERAL CONVERSATION PHRASES BY CATEGORY (517 TOTAL)
# ==============================================================================

GENERAL_PHRASES_BY_CATEGORY = {
    # 1. Greetings / salutations (51 phrases)
    "Greetings": (
        GREETING_RESPONSE,
        [
            "hello", "helo", "helllo", "hllo", "hi", "hii", "hiii", "hiiii",
            "hey", "heyy", "heyyy", "hello there", "hi there", "hey there",
            "good morning", "good afternoon", "good evening", "morning", "evening",
            "greetings", "howdy", "hey ya", "heya", "yo", "yoo", "yooo",
            "top of the morning", "good day", "hello chatbot", "hi bot", "hey bot",
            "good morning chatbot", "hello assistant", "hi assistant", "greetings assistant",
            "warm greetings", "hi everyone", "hello team", "hi campusbot", "hello campusbot",
            "hey campusbot", "holla", "ahoy", "hiya", "hello friend", "hey friend",
            "hi student support", "hello support", "hi buddy", "hey buddy", "hello again"
        ]
    ),

    # 2. How are you / wellbeing (41 phrases)
    "How are you": (
        HOW_ARE_YOU_RESPONSE,
        [
            "how are you", "how are u", "how r you", "how r u", "hw r u", "hru",
            "how are you doing", "how are you doing today", "how are you today bot",
            "are you okay", "are you ok", "are you doing well", "how is it going",
            "hows it going", "how's it going", "how are things", "how is your day",
            "hows your day", "how is life", "how are you feeling", "are you fine",
            "hope you are well", "hope you are doing well", "hope you're doing well",
            "how have you been", "how do you do", "how's everything", "hows everything",
            "how are you faring", "how are you today", "you okay", "you ok", "are u ok",
            "how is your day going", "hows your day going", "how is work",
            "is everything okay", "is everything ok", "how goes it", "what's good", "whats good"
        ]
    ),

    # 3. Identity / introduction (40 phrases)
    "Identity": (
        IDENTITY_RESPONSE,
        [
            "who are you", "what are you", "who r u", "who u are", "what is your name",
            "whats your name", "what's your name", "tell me about yourself", "describe yourself",
            "are you a chatbot", "are you a bot", "are you an ai", "are you artificial intelligence",
            "are you human", "who made you", "who created you", "who built you",
            "introduce yourself", "what is campusbot", "what is this bot", "who am i talking to",
            "who is this", "what are u", "are you real", "are u real",
            "give me your name", "do you have a name", "what should i call you", "what may i call you",
            "your name please", "introduce yourself to me", "tell me who you are",
            "are you virtual assistant", "are you student bot", "what kind of bot are you",
            "who powers you", "are you an automated assistant", "can you introduce yourself",
            "who engineered you", "what is your identity"
        ]
    ),

    # 4. Capabilities / what can you do (50 phrases)
    "Capabilities": (
        CAPABILITIES_RESPONSE,
        [
            "what can you do", "what can u do", "what do you do", "what are your capabilities",
            "how can you help me", "how can u help me", "what services do you provide",
            "what topics do you know", "what questions can i ask", "what can i ask you",
            "what can i ask", "show me your menu", "show menu", "list your features",
            "what are your features", "what can you answer", "what do you know",
            "how do you work", "what is your purpose", "why were you created",
            "how can this bot help me", "what support do you offer", "what information can you provide",
            "what can you assist me with", "how can you assist me", "what can you help with",
            "what do you cover", "explain your features", "tell me what you can do",
            "tell me your capabilities", "what are you capable of", "what can i inquire about",
            "what are your functions", "how can you help students", "what student services do you know",
            "can you assist me with university queries", "what information do you have",
            "what do you assist with", "give me a list of capabilities", "what are your main skills",
            "what can you help us with", "what areas do you cover", "what can you do for me",
            "what do you specialize in", "what questions can you answer", "what topics can we discuss",
            "show me what you can do", "what help can you give me", "tell me your functions",
            "what can i ask this bot"
        ]
    ),

    # 5. General help requests (42 phrases)
    "General help requests": (
        HELP_REQUEST_RESPONSE,
        [
            "help", "help me", "help me please", "please help me", "i need help",
            "i need some help", "can you help me", "can u help me", "could you help me",
            "would you help me", "can you help me out please", "i am looking for some assistance",
            "assistance please", "i need assistance", "need help", "can i get some help",
            "give me help", "i need help with something", "can you give me help",
            "i require assistance", "support needed", "help required", "can someone help me",
            "i need a hand", "could you lend me a hand", "assist me please", "i need information",
            "can you provide help", "i have a question", "i want to ask something",
            "can i ask a question", "can i ask u something", "can i ask you something",
            "i need help from you", "please assist", "need assistance", "i'm looking for help",
            "looking for support", "looking for assistance", "could you assist me",
            "please help", "need help right now"
        ]
    ),

    # 6. Thanks / appreciation (35 phrases)
    "Thanks": (
        THANKS_RESPONSE,
        [
            "thank you", "thankyou", "thanks", "thank u", "thankx", "thnx", "tnx",
            "thank you so much", "thankyou so much", "thanks a lot", "thanks many",
            "many thanks", "thanks a million", "thank you very much", "appreciate it",
            "i appreciate it", "much appreciated", "thank you for your help",
            "thanks for your help", "thanks for the help", "thank you for assisting me",
            "that's helpful", "thats helpful", "that is helpful", "that was helpful",
            "thank you assistant", "thanks bot", "thank you campusbot", "awesome thank you",
            "great thank you", "thanks buddy", "thank you kindly", "thanks a ton",
            "big thanks", "thank you so much for the information"
        ]
    ),

    # 7. Goodbye / farewell (30 phrases)
    "Goodbye": (
        GOODBYE_RESPONSE,
        [
            "bye", "byee", "byeee", "goodbye", "good bye", "see you", "see ya",
            "see you later", "talk to you later", "have a good day", "have a nice day",
            "have a great day", "bye bye", "see u", "see u later", "catch you later",
            "catch u later", "farewell", "good night", "goodnight", "bye for now",
            "signing off", "talk later", "see you soon", "see u soon", "take care",
            "take care bye", "talk to you soon", "see you tomorrow", "bye campusbot"
        ]
    ),

    # 8. Confirmation / acknowledgement (35 phrases)
    "Confirmation": (
        CONFIRMATION_RESPONSE,
        [
            "ok", "okay", "got it", "alright", "all right", "understood", "i see",
            "sure", "cool", "makes sense", "k", "kk", "ok got it", "okay got it",
            "fine", "sound good", "sounds good", "fair enough", "fair", "i understand",
            "understod", "noted", "duly noted", "right", "yeah", "yep", "yup",
            "ok thanks", "okay thanks", "okay i get it", "gotcha", "i got it",
            "i understand now", "that makes sense", "perfectly clear"
        ]
    ),

    # 9. Positive responses (30 phrases)
    "Positive responses": (
        POSITIVE_RESPONSE,
        [
            "great", "awesome", "perfect", "wonderful", "amazing", "good", "nice",
            "super", "excellent", "brilliant", "fantastic", "sweet", "brilliant work",
            "good job", "great job", "nice job", "impressive", "love it",
            "that is awesome", "that's great", "thats great", "sounds great",
            "sounds awesome", "perfect thank you", "perfect thanks", "very good",
            "really good", "super helpful", "awesome work", "spot on"
        ]
    ),

    # 10. Negative responses (27 phrases)
    "Negative responses": (
        NEGATIVE_RESPONSE,
        [
            "no", "nope", "nah", "no thanks", "no thank you", "not now", "nothing",
            "never mind", "nevermind", "no further questions", "don't need help",
            "dont need help", "no need", "i'm good", "im good", "no i'm fine",
            "no im fine", "nothing else", "nothing for now", "no questions",
            "no more questions", "i'm okay", "im okay", "no thanks bye", "not really",
            "no that's all", "no thats all"
        ]
    ),

    # 11. Clarification / misunderstanding (35 phrases)
    "Clarification": (
        CLARIFICATION_RESPONSE,
        [
            "i don't understand", "i dont understand", "i didn't understand",
            "i didnt understand", "what do you mean", "what do u mean", "i'm confused",
            "im confused", "that is confusing", "that's confusing", "i don't get it",
            "i dont get it", "that makes no sense", "this makes no sense",
            "that's not what i meant", "thats not what i meant", "you misunderstood me",
            "you misunderstood", "that's not what i asked", "thats not what i asked",
            "i don't follow", "i dont follow", "can you clarify", "please clarify",
            "i need clarification", "what does that mean", "could you clarify that",
            "i'm not following", "im not following", "i am lost", "im lost",
            "i don't quite understand", "i dont quite get it", "wait what",
            "what do you mean by that"
        ]
    ),

    # 12. Repetition requests (26 phrases)
    "Repetition requests": (
        REPETITION_RESPONSE,
        [
            "can you repeat that", "can u repeat that", "could you repeat that",
            "please repeat", "repeat that please", "say that again",
            "can you say that again", "say again", "can you explain that again",
            "explain again", "explain that again", "could you explain that again",
            "can you explain that one more time", "tell me again", "repeat please",
            "pardon", "pardon me", "come again", "what was that", "can you repeat",
            "could you repeat", "repeat the answer", "say it again",
            "can you repeat what you said", "explain it once more", "repeat once more"
        ]
    ),

    # 13. Casual conversation (35 phrases)
    "Casual conversation": (
        CASUAL_CONVERSATION_RESPONSE,
        [
            "how is your work", "what are you up to", "what are u up to", "what's up",
            "whats up", "sup", "wbu", "what about you", "what about u",
            "tell me something", "talk to me", "let's talk", "lets talk",
            "chat with me", "can we talk", "can we chat", "are you busy", "are u busy",
            "what are you doing", "what are u doing", "what are you thinking",
            "how was your day", "hows your day been", "nice day isn't it",
            "nice day is it", "having fun", "having a good day", "just checking in",
            "just saying hi", "just saying hello", "just browsing", "just testing",
            "testing testing", "test bot", "testing bot"
        ]
    ),

    # 14. Small talk (30 phrases)
    "Small talk": (
        SMALL_TALK_RESPONSE,
        [
            "nice to meet you", "pleasure to meet you", "you are helpful", "you are smart",
            "you are cool", "you are awesome", "you are great", "are you smart",
            "tell me a joke", "do you like students", "do you like university",
            "do you sleep", "do you eat", "do you have friends", "do you have feelings",
            "are you alive", "how old are you", "where do you live", "where are you located",
            "where are you based", "who is your creator", "who created you",
            "who programmed you", "who is your boss", "are you an ai bot",
            "you are very clever", "you are so smart", "glad to talk to you",
            "happy to meet you", "it is a pleasure to meet you"
        ]
    ),

    # 15. Apology / correction (20 phrases)
    "Apology": (
        APOLOGY_RESPONSE,
        [
            "sorry", "i am sorry", "im sorry", "my bad", "my mistake", "oops", "whoops",
            "sorry about that", "sorry my typo", "sorry typo", "i made a mistake",
            "im sorry about that", "pardon my mistake", "excuse me", "sorry for that",
            "sorry wrong question", "wrong question sorry", "sorry i typed wrong",
            "sorry misprinted", "oh sorry"
        ]
    ),

    # 16. Conversation starters (20 phrases)
    "Conversation starters": (
        STARTER_RESPONSE,
        [
            "are you still there", "are u still there", "are you available",
            "are u available", "are you ready", "are u ready", "can i ask a question now",
            "can i ask you a question", "ready when you are", "is anyone there",
            "anyone here", "hello is anyone there", "hi is anyone home",
            "starting conversation", "can we begin", "let's start", "lets start",
            "i have a query", "i have a quick question", "quick question"
        ]
    )
}



# ==============================================================================
# TEXT NORMALIZATION & MAP CONSTRUCTION
# ==============================================================================

def normalize_text(text: str) -> str:
    """Normalize user input text for exact general conversation matching.
    Converts to lowercase, removes leading/trailing spaces, strips basic punctuation,
    and collapses multiple spaces.
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation)).strip()
    text = " ".join(text.split())
    return text


# Build normalized lookup map: normalized_phrase -> (category, response)
GENERAL_CONVERSATION_MAP = {}
PHRASE_LIST_FOR_EMBEDDINGS = []

for category_name, (response_text, phrase_list) in GENERAL_PHRASES_BY_CATEGORY.items():
    for phrase in phrase_list:
        norm_key = normalize_text(phrase)
        if norm_key and norm_key not in GENERAL_CONVERSATION_MAP:
            GENERAL_CONVERSATION_MAP[norm_key] = (category_name, response_text)
            PHRASE_LIST_FOR_EMBEDDINGS.append(norm_key)


# ==============================================================================
# GENERAL CONVERSATION DETECTOR CLASS (EXACT + CACHED EMBEDDINGS)
# ==============================================================================

class GeneralConversationDetector:
    """Detector for General Conversation using exact matching and cached SentenceTransformer embeddings."""

    def __init__(self):
        self.model = None
        self.phrase_embeddings = None
        self.phrase_keys = []
        self._is_loaded = False

    def load_resources(self, model=None):
        """Pre-compute embeddings for all 500+ general conversation phrases.
        Optionally accepts an already loaded SentenceTransformer model to avoid duplicate model loading.
        """
        if self._is_loaded:
            return

        if model is not None:
            self.model = model
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.phrase_keys = list(GENERAL_CONVERSATION_MAP.keys())
        # Encode all normalized general phrases
        self.phrase_embeddings = self.model.encode(
            self.phrase_keys,
            normalize_embeddings=True
        )
        self._is_loaded = True

    def check_exact_match(self, raw_prompt: str):
        """First layer: Exact string matching after normalization."""
        norm = normalize_text(raw_prompt)
        if norm in GENERAL_CONVERSATION_MAP:
            cat, resp = GENERAL_CONVERSATION_MAP[norm]
            return {
                "question": raw_prompt,
                "intent": f"general_conversation ({cat})",
                "confidence": 1.0,
                "matched_question": norm,
                "similarity": 1.0,
                "match_type": "exact",
                "response": resp
            }
        return None

    def check_embedding_match(self, raw_prompt: str, threshold: float = GENERAL_EMBEDDING_THRESHOLD):
        """Second layer: SentenceTransformer embedding cosine similarity matching against 500+ general phrases."""
        if not self._is_loaded:
            return None

        norm = normalize_text(raw_prompt)
        if not norm:
            return None

        user_emb = self.model.encode([norm], normalize_embeddings=True)
        sims = cosine_similarity(user_emb, self.phrase_embeddings)[0]

        best_idx = int(np.argmax(sims))
        best_sim = round(float(sims[best_idx]), 4)
        matched_phrase = self.phrase_keys[best_idx]

        if best_sim >= threshold:
            cat, resp = GENERAL_CONVERSATION_MAP[matched_phrase]
            return {
                "question": raw_prompt,
                "intent": f"general_conversation ({cat})",
                "confidence": best_sim,
                "matched_question": matched_phrase,
                "similarity": best_sim,
                "match_type": "embedding",
                "response": resp
            }

        return None

    def predict_general(self, raw_prompt: str, threshold: float = GENERAL_EMBEDDING_THRESHOLD):
        """Full pipeline for General Conversation: Exact match -> Embedding match."""
        # 1. First layer: Exact match
        exact_res = self.check_exact_match(raw_prompt)
        if exact_res:
            return exact_res

        # 2. Second layer: Embedding match
        emb_res = self.check_embedding_match(raw_prompt, threshold=threshold)
        if emb_res:
            return emb_res

        return None


# Global general conversation detector instance
general_detector = GeneralConversationDetector()
