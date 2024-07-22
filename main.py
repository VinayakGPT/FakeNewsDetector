import streamlit as st
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
with open('trained_model.sav', 'rb') as f:
    loaded_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# Preprocessing function
def preprocess(title, author, text):
    def clean_text(text):
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return text

    combined_text = f"{title} {author} {text}"
    cleaned_text = clean_text(combined_text)
    return vectorizer.transform([cleaned_text])


# Streamlit app
st.title("Fake News Detection")

title = st.text_input("Title")
author = st.text_input("Author")
text = st.text_area("Text")

if st.button("Predict"):
    X_new = preprocess(title, author, text)
    prediction = loaded_model.predict(X_new)
    if prediction[0] == 0:
        st.success('News is real!')
    else:
        st.error('News is fake!')
