import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download required nltk data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    """
    Basic NLP preprocessing: 
    - Lowercase
    - Remove punctuation
    - Remove numbers
    - Tokenization
    - Stopword removal
    - Lemmatization
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return " ".join(tokens)

def get_suspicious_phrases(text, top_n=5):
    # This is a placeholder for actual explainability logic
    # In a real transformer model, we'd use attention weights or SHAP
    # For now, let's just return some words that might be indicators (mock)
    words = text.split()
    # Simple logic for demo: words with lots of capital letters or exclamation marks
    # (after re-visiting original text)
    return words[:top_n]
