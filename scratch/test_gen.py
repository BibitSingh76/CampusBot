import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from streamlit_app import check_general_conversation, GENERAL_CONVERSATION_MAP, GENERAL_PHRASES_BY_CATEGORY
from backend.app.classifier import classifier

print("Loading classifier resources...")
classifier.load_resources()

print(f"\nTotal unique phrases in GENERAL_CONVERSATION_MAP: {len(GENERAL_CONVERSATION_MAP)}")

categories_summary = {}
for cat, (resp, phrases) in GENERAL_PHRASES_BY_CATEGORY.items():
    categories_summary[cat] = len(phrases)
    print(f"  - {cat}: {len(phrases)} phrases")

print("\n--- RUNNING REQUIRED TESTS ---")

test_cases = [
    "hello",
    "helo",
    "hi",
    "hii",
    "hey",
    "how are you",
    "how r u",
    "who are you",
    "what is your name",
    "thank you",
    "thanks",
    "bye",
    "goodbye",
    "How do I register for a course?",
    "What is the capital of France?",
    "Which courses are available?",
    "How can I contact the scholarship office?",
    "The student portal is not working.",
    "hullo"
]

for prompt in test_cases:
    match = check_general_conversation(prompt)
    if match:
        print(f"[GENERAL MATCH] '{prompt}' => Intent: {match['intent']} | Response: {match['response']}")
    else:
        ml_res = classifier.predict(prompt)
        print(f"[ML FALLTHROUGH] '{prompt}' => Intent: {ml_res['intent']} | Confidence: {ml_res['confidence']} | Matched Q: {ml_res['matched_question']} | Response: {ml_res['response']}")
