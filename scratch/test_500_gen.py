import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from general_conversation import (
    general_detector,
    GENERAL_CONVERSATION_MAP,
    GENERAL_PHRASES_BY_CATEGORY,
    GENERAL_EMBEDDING_THRESHOLD
)
from backend.app.classifier import classifier

print("1. Loading classifier & general conversation resources...")
classifier.load_resources()
general_detector.load_resources(model=classifier.model)

print(f"\n2. Total unique general phrases: {len(GENERAL_CONVERSATION_MAP)}")
print(f"3. General embedding threshold: {GENERAL_EMBEDDING_THRESHOLD}")

print("\n--- CATEGORY SUMMARY ---")
total_count = 0
for idx, (cat_name, (resp, phrases)) in enumerate(GENERAL_PHRASES_BY_CATEGORY.items(), 1):
    total_count += len(phrases)
    print(f"  {idx:2d}. {cat_name:<35} : {len(phrases):2d} phrases")
print(f"TOTAL PHRASES IN DICTIONARY: {total_count}")

print("\n--- TEST GENERAL CONVERSATION (EXACT & EMBEDDING) ---")

general_test_prompts = [
    # Exact & spelling / formatting variations
    "hello",
    "HELO",
    " hi ",
    "Hey!",
    "how are you?",
    "how r u",
    "who are you?",
    "what can you do?",
    "can you help me?",
    "thank you",
    "thanks!",
    "can you explain that again?",
    "bye",
    "  HELLO!!!  ",
    "thnx",
    "are you still there",
    "can I ask you something?",
    "I need some help",
    "Never mind",
    "That's not what I meant",
    # Embedding variations (not exact string, but high similarity)
    "good morning chatbot",
    "can you explain that one more time",
    "how are you today bot"
]

for prompt in general_test_prompts:
    res = general_detector.predict_general(prompt)
    if res:
        print(f"[MATCH - {res['match_type'].upper()}] '{prompt}' => [{res['intent']}] Sim: {res['similarity']} | Response: {res['response']}")
    else:
        print(f"[NO GENERAL MATCH] '{prompt}'")

print("\n--- TEST UNIVERSITY QUESTIONS (MUST PASS THROUGH TO ML CLASSIFIER) ---")

university_test_prompts = [
    "How do I register for a course?",
    "How can I apply for a scholarship?",
    "Where can I find the student portal?",
    "Which courses are available?",
    "What are the library hours?",
    "How do I contact academic advising?",
    "The student portal is not working.",
    "How much is tuition?",
    "When does registration start?"
]

for prompt in university_test_prompts:
    gen_res = general_detector.predict_general(prompt)
    if gen_res:
        print(f"[ERROR - INCORRECT GENERAL MATCH] '{prompt}' => Matched general intent: {gen_res['intent']} (Sim: {gen_res['similarity']})")
    else:
        ml_res = classifier.predict(prompt)
        print(f"[SUCCESS - ML PIPELINE] '{prompt}' => ML Intent: {ml_res['intent']} (Conf: {ml_res['confidence']}) | Matched Q: {ml_res['matched_question']}")
