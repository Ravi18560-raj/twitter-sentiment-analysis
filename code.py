# Installing libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Cleaning: Remove URLs, mentions (@), hashtags (#), numbers, and punctuation
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URLs
    text = re.sub(r'@\w+|#\w+', '', text)             # @mentions and #hashtags
    text = re.sub(r'[^\w\s]', '', text)              # Punctuation
    text = re.sub(r'\d+', '', text)                  # Numbers
    text = text.strip()
    
    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Remove Stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Return cleaned sentence string
    return " ".join(filtered_tokens)

# Load dataset
df = pd.read_csv('twitter_data.csv')

# Create the clean_text column
df['clean_text'] = df['tweet'].apply(preprocess_text)

# Select only clean_text and category
final_df = df[['clean_text', 'category']]

# Save or display
final_df.to_csv('cleaned_twitter_data.csv', index=False)
print(final_df.head())

