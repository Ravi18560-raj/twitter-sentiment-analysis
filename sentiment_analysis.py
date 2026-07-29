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

# Converting text into features (TF-IDF)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 1. Load the cleaned dataset created by code.py
df = pd.read_csv("cleaned_twitter_data.csv")

# Handle any missing values in clean_text
df["clean_text"] = df["clean_text"].fillna("")

# 2. Extract features (X) and targets (y)
X_raw = df["clean_text"]
y = df["category"]

# 3. Initialize TF-IDF Vectorizer
# - max_features: limits vocabulary to top 5,000 words
# - ngram_range=(1,2): includes unigrams ("good") and bigrams ("not good")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# 4. Transform text into numerical feature matrix
X = tfidf.fit_transform(X_raw)

print("TF-IDF Matrix Shape:", X.shape)
# Output shape will be: (number_of_rows, 5000)

# 5. Split into train and test sets for model training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
