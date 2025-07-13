import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# === Load saved model and vectorizer ===
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    st.error(" Error loading model or vectorizer.")
    st.exception(e)
    st.stop()

# Stopwords
stop_words = set(stopwords.words('english'))

# === Preprocessing Function ===
def preprocess(text):
    text = text.lower()
    words = word_tokenize(text)
    cleaned = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(cleaned)

# === Streamlit UI ===
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title(" Fake News Detector")
st.write("Paste a news headline or article below to check if it's **Real** or **Fake**.")

# User input
user_input = st.text_area("Enter News Text")

# On button click
if st.button("Check"):
    if user_input.strip() == "":
        st.warning(" Please enter some text before clicking Check.")
    else:
        # Preprocess and predict
        cleaned = preprocess(user_input)
        vector = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vector)[0]

        # Show result
        if prediction == 1:
            st.success(" This looks like **Real News**.")
        else:
            st.error(" This appears to be **Fake News**.")
