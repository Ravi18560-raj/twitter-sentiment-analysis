# Installing libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def clean_and_tokenize(text):
    """
    Cleans the text by removing links, special characters, and numbers,
    then tokenizes it and removes stopwords.
    """
    if not isinstance(text, str):
        return ""
# Preprocess text
# 1. Cleaning
    text = text.lower() # Convert to lowercase
    text = re.sub(r"http\s+|www\S+|https\S+", "", text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text) # Remove mentions (@) and hashtags (#)
    text = re.sub(r"[^\w\s]", "", text) # Remove punctuation
    text = re.sub(r"\d+", "", text) # Remove numbers
    text = text.strip() # Remove leading/trailing whitespace
# 2. Tokenisation    
    tokens = word_tokenize(text)
# 3. Stopwords removal    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# Load the dataset
df = pd.read_csv('twitter_data.csv')

