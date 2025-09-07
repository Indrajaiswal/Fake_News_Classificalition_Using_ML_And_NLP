import streamlit as st
import pickle
import string
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ----------------------------
# Initialize Stemmer
# ----------------------------
ps = PorterStemmer()

# ----------------------------
# Text Preprocessing Function (no nltk.word_tokenize)
# ----------------------------
def transform_text(text):
    text = text.lower()  # lowercase
    
    # Split using regex to handle punctuation and spaces
    text = re.findall(r'\b\w+\b', text)
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
    
    # Apply stemming
    text = [ps.stem(word) for word in text]
    
    return " ".join(text)

# ----------------------------
# Load Vectorizer and Model
# ----------------------------
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"❌ Error loading model or vectorizer: {e}")
    st.stop()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection Project")
st.write("""
Enter any news article or content below, and the model will classify it as **True News** or **Fake News**.
""")

input_content = st.text_area("Enter the news article/content here:")

if st.button("Predict"):
    if input_content.strip() == "":
        st.warning("⚠️ Please enter a valid news article or content.")
    else:
        transformed_content = transform_text(input_content)
        vector_input = tfidf.transform([transformed_content])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.success("✅ This news is **TRUE**.")
        else:
            st.error("⚠️ This news is **FAKE**.")
