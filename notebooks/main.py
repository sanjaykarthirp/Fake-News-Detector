import pandas as pd
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer from correct relative path
try:
    model = pickle.load(open('D:/fake-news-project/models/model.pkl', 'rb'))
    vectorizer = pickle.load(open('D:/fake-news-project/models/vectorizer.pkl', 'rb'))
except Exception as e:
    print("Error loading model or vectorizer:", e)
    exit()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    words = word_tokenize(text)
    cleaned = [word for word in words if word.isalpha()]
    return ' '.join(cleaned)

# --- Run the prediction ---
while True:
    user_input = input("\nEnter news statement (or type 'exit' to quit):\n> ")
    if user_input.lower() == 'exit':
        break

    cleaned_text = preprocess(user_input)
    vector = vectorizer.transform([cleaned_text]).toarray()
    prob_real = model.predict_proba(vector)[0][1]

    if prob_real > 0.6:
        print("This appears to be **Real News**.")
    elif prob_real < 0.4:
        print("This appears to be **Fake News**.")
    else:
        print("This is uncertain. Try giving more context.")
