import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ----------------------------
# Ensure NLTK data is available
# ----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ----------------------------
# Initialize Stemmer
# ----------------------------
ps = PorterStemmer()

# ----------------------------
# Text Preprocessing Function
# ----------------------------
def transform_text(text):
    text = text.lower()  # Lowercase
    text = nltk.word_tokenize(text)  # Tokenize
    
    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    y = [i for i in y if i not in stop_words and i not in string.punctuation]
    
    # Stemming
    y = [ps.stem(i) for i in y]
    
    return " ".join(y)

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

# Input content
input_content = st.text_area("Enter the news article/content here:")

if st.button("Predict"):
    if input_content.strip() == "":
        st.warning("⚠️ Please enter a valid news article or content.")
    else:
        # Preprocess input
        transformed_content = transform_text(input_content)
        
        # Vectorize input
        vector_input = tfidf.transform([transformed_content])
        
        # Predict
        result = model.predict(vector_input)[0]
        
        # Display result
        if result == 1:
            st.success("✅ This news is **TRUE**.")
        else:
            st.error("⚠️ This news is **FAKE**.")
