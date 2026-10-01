import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from streamlit_app import check_general_conversation, GENERAL_CONVERSATION_MAP, GENERAL_PHRASES_BY_CATEGORY

print(f"Total unique phrases in GENERAL_CONVERSATION_MAP: {len(GENERAL_CONVERSATION_MAP)}")

print("\n--- CATEGORIES & PHRASE COUNTS ---")
for cat, (resp, phrases) in GENERAL_PHRASES_BY_CATEGORY.items():
    print(f"  - {cat}: {len(phrases)} phrases")

print("\n--- TEST MATCHING GENERAL CONVERSATIONS ---")
test_general = [
    "hello", "HELO", " hi ", "hii", "hey!", "how are you?", "how r u",
    "who are you", "what is your name", "thank you", "thanks!", "bye", "goodbye",
    "good morning", "good evening", "what can you do", "appreciate it", "see you"
]

for prompt in test_general:
    match = check_general_conversation(prompt)
    if match:
        print(f"MATCH: '{prompt}' -> [{match['intent']}] {match['response']}")
    else:
        print(f"NO MATCH: '{prompt}'")

print("\n--- TEST NON-MATCHING / UNIVERSITY QUESTIONS ---")
test_university = [
    "How do I register for a course?",
    "How can I apply for a scholarship?",
    "Where can I find the student portal?",
    "What are the library hours?",
    "How do I contact academic advising?",
    "Which courses are available?",
    "How can I contact the scholarship office?",
    "The student portal is not working.",
    "hullo"
]

for prompt in test_university:
    match = check_general_conversation(prompt)
    if match:
        print(f"INCORRECT MATCH: '{prompt}' -> {match['intent']}")
    else:
        print(f"SAFE FALLTHROUGH TO ML: '{prompt}'")
