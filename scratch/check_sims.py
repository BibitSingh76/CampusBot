import sys
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from general_conversation import (
    general_detector,
    GENERAL_CONVERSATION_MAP,
    GENERAL_EMBEDDING_THRESHOLD,
    normalize_text
)
from backend.app.classifier import classifier

classifier.load_resources()
general_detector.load_resources(model=classifier.model)

phrase_keys = list(GENERAL_CONVERSATION_MAP.keys())
phrase_embs = general_detector.phrase_embeddings

test_prompts = [
    # Natural general embedding variations
    "good morning chatbot",
    "can you explain that one more time",
    "how are you today bot",
    "never mind",
    "nevermind",
    "can you help me out please",
    "i am looking for some assistance",
    # University questions
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

print(f"{'PROMPT':<40} | {'BEST MATCHED GENERAL PHRASE':<35} | SIMILARITY")
print("-" * 88)

for prompt in test_prompts:
    norm = normalize_text(prompt)
    user_emb = classifier.model.encode([norm], normalize_embeddings=True)
    sims = cosine_similarity(user_emb, phrase_embs)[0]
    best_idx = int(np.argmax(sims))
    best_sim = round(float(sims[best_idx]), 4)
    best_phrase = phrase_keys[best_idx]
    print(f"{prompt:<40} | {best_phrase:<35} | {best_sim}")
