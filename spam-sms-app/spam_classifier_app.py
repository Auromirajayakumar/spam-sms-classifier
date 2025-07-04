import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import pandas as pd

nltk.download('stopwords')

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Streamlit UI
st.set_page_config(page_title="Spam SMS Classifier", layout="centered")
st.title("📩 Spam SMS Classifier")
st.markdown("Enter an SMS message below to check if it's **Spam** or **Not Spam**")

user_input = st.text_area("💬 Enter SMS Text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        cleaned = clean_text(user_input)
        result = model.predict([cleaned])[0]
        if result == 1:
            st.error("🚨 This is *SPAM*!")
        else:
            st.success("✅ This is *NOT SPAM*!")
