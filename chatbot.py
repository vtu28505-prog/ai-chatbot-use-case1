import re
import random
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)


# =========================================================
# GENERAL CHATBOT KNOWLEDGE BASE
# =========================================================

intents = {

    "greeting": {
        "patterns": [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "how are you doing"
        ],
        "responses": [
            "Hello! How can I help you?",
            "Hi! What can I do for you?",
            "Hey! Nice to talk to you."
        ]
    },

    "name": {
        "patterns": [
            "what is your name",
            "who are you",
            "tell me your name",
            "what should I call you",
            "who am I talking to"
        ],
        "responses": [
            "I'm a basic AI chatbot designed to have conversations with you.",
            "You can call me SmartBot.",
            "I'm your virtual assistant."
        ]
    },

    "how_are_you": {
        "patterns": [
            "how are you",
            "are you okay",
            "how are you feeling",
            "are you doing well"
        ],
        "responses": [
            "I'm doing great! Thanks for asking.",
            "I'm fine and ready to help!",
            "Doing well! What would you like to talk about?"
        ]
    },

    "thanks": {
        "patterns": [
            "thank you",
            "thanks",
            "thank you so much",
            "thanks a lot",
            "i appreciate it"
        ],
        "responses": [
            "You're welcome!",
            "Happy to help!",
            "Anytime!",
            "Glad I could help."
        ]
    },

    "help": {
        "patterns": [
            "help me",
            "what can you do",
            "what do you do",
            "how can you help me",
            "what are your capabilities"
        ],
        "responses": [
            "I can have basic conversations, answer common questions, and respond to different types of messages.",
            "You can ask me questions, have a conversation, or ask for general information."
        ]
    },

    "weather": {
        "patterns": [
            "how is the weather",
            "what is the weather",
            "is it raining",
            "is it hot today",
            "is it cold today"
        ],
        "responses": [
            "I don't have access to live weather data, but you can ask me general questions about weather.",
            "I can't check live weather yet, but I can discuss weather-related topics."
        ]
    },

    "time": {
        "patterns": [
            "what time is it",
            "tell me the time",
            "current time",
            "what is the time"
        ],
        "responses": [
            "You can check your device clock for the current time.",
            "I don't have a live clock in this basic version."
        ]
    },

    "joke": {
        "patterns": [
            "tell me a joke",
            "make me laugh",
            "say something funny",
            "tell me something funny"
        ],
        "responses": [
            "Why do programmers prefer dark mode? Because light attracts bugs! 😂",
            "Why was the computer cold? It left its Windows open! 😂",
            "I told my computer I needed a break... now it won't stop sending me vacation ads."
        ]
    },

    "sad": {
        "patterns": [
            "i am sad",
            "i feel sad",
            "i am unhappy",
            "i feel terrible",
            "i feel bad",
            "i am feeling down"
        ],
        "responses": [
            "I'm sorry you're feeling this way. Want to talk about it?",
            "That sounds difficult. I'm here to listen.",
            "I hope things get better. You can tell me what's bothering you."
        ]
    },

    "happy": {
        "patterns": [
            "i am happy",
            "i feel happy",
            "i am excited",
            "i feel great",
            "today is a good day"
        ],
        "responses": [
            "That's great to hear! 😊",
            "I'm glad you're feeling good!",
            "That's wonderful! What happened?"
        ]
    },

    "goodbye": {
        "patterns": [
            "bye",
            "goodbye",
            "see you",
            "see you later",
            "exit",
            "quit"
        ],
        "responses": [
            "Goodbye! Have a great day!",
            "See you later!",
            "Bye! Take care!"
        ]
    }
}


# =========================================================
# PREPARE TRAINING DATA
# =========================================================

patterns = []
labels = []

for intent, data in intents.items():

    for pattern in data["patterns"]:

        patterns.append(pattern)
        labels.append(intent)


# =========================================================
# TF-IDF MODEL
# =========================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2)
)

pattern_vectors = vectorizer.fit_transform(patterns)


# =========================================================
# CHATBOT ENGINE
# =========================================================

def chatbot(user_input):

    # Clean input
    user_input = user_input.lower().strip()

    if not user_input:
        return "Please type something so I can respond."

    # Convert user sentence into vector
    user_vector = vectorizer.transform([user_input])

    # Calculate similarity
    similarities = cosine_similarity(
        user_vector,
        pattern_vectors
    )[0]

    # Find closest sentence
    best_index = similarities.argmax()

    best_score = similarities[best_index]

    # Confidence threshold
    if best_score < 0.25:

        return (
            "I'm not sure I understand that yet. "
            "Could you explain it in another way?"
        )

    # Find corresponding intent
    intent = labels[best_index]

    # Generate response
    return random.choice(
        intents[intent]["responses"]
    )


# =========================================================
# TERMINAL CHAT
# =========================================================

print("=" * 60)
print("🤖 GENERAL AI CHATBOT")
print("=" * 60)

print("Type 'bye' to end the conversation.")
print()


while True:

    user = input("You: ")

    response = chatbot(user)

    print("Bot:", response)

    if user.lower().strip() in [
        "bye",
        "goodbye",
        "exit",
        "quit"
    ]:
        break
