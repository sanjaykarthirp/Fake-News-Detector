# === NLP SETUP ===
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# === MACHINE LEARNING LIBRARIES ===
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# === LOAD DATASETS ===
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

# Add labels: 0 = Fake, 1 = Real
fake_df['label'] = 0
true_df['label'] = 1

# Combine and shuffle
df = pd.concat([fake_df, true_df]).sample(frac=1).reset_index(drop=True)
print(" Data Loaded Successfully!")

# === TEXT PREPROCESSING ===
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    words = word_tokenize(text)
    cleaned = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(cleaned)

# Apply cleaning to all text
df['clean_text'] = df['text'].apply(preprocess)
print(df[['text', 'clean_text']].head(10))

# === TF-IDF FEATURE EXTRACTION ===
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['label'].values

print(" Shape of feature matrix:", X.shape)
print(" Example TF-IDF vector:", X[0])

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(" Training shape:", X_train.shape)
print(" Testing shape:", X_test.shape)

print(df['label'].value_counts())
df['clean_text'] = df['text'].apply(preprocess)

# === MODEL TRAINING ===
model = MultinomialNB()
model.fit(X_train, y_train)
print(" Model training complete!")

# === EVALUATION ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f" Accuracy: {acc:.2f}")
print(" Classification Report:")
print(classification_report(y_test, y_pred))


def predict_news(text):
    cleaned = preprocess(text)
    vector = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]
    return " Real News " if prediction == 1 else " Fake News "

# Test
sample_news = "NASA confirms water found on the moon's surface."
print("Prediction ", predict_news(sample_news))

import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

