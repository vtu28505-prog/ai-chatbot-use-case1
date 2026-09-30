# AI Chatbot - Task 1

A basic AI chatbot developed using Python, TF-IDF vectorization, and cosine similarity.

## Features

- Greeting detection
- Name-related questions
- General conversation
- Help responses
- Weather-related responses
- Time-related responses
- Jokes
- Happy and sad responses
- Goodbye detection

## Technologies Used

- Python
- NLTK
- Scikit-learn
- TF-IDF
- Cosine Similarity

## How It Works

The chatbot stores predefined patterns and responses under different intents.

When a user enters a question, the input is converted into a TF-IDF vector. Cosine similarity is then used to compare the input with the stored patterns.

The chatbot selects the most similar pattern and returns a response from the corresponding intent.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
