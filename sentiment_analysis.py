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

# Training the model (Naive Bayes, Logistic Regression, LSTM)
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# 1. Load Data
df = pd.read_csv("cleaned_twitter_data.csv")
df["clean_text"] = df["clean_text"].fillna("")

# Encode target categories to integers (0, 1, 2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df["category"])

# ==========================================
# PART 1: TF-IDF Models (Naive Bayes & Logistic Regression)
# ==========================================
print("--- Training TF-IDF Models ---")

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df["clean_text"])

# Train-Test Split for TF-IDF
X_train_tf, X_test_tf, y_train, y_test = train_test_split(
    X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 1. Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tf, y_train)
nb_preds = nb_model.predict(X_test_tf)

print("\n[1] Naive Bayes Accuracy:", accuracy_score(y_test, nb_preds))
print(
    classification_report(
        y_test, nb_preds, target_names=label_encoder.classes_
    )
)

# 2. Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tf, y_train)
lr_preds = lr_model.predict(X_test_tf)

print("\n[2] Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))
print(
    classification_report(
        y_test, lr_preds, target_names=label_encoder.classes_
    )
)

# ==========================================
# PART 2: Sequential Deep Learning Model (LSTM)
# ==========================================
print("\n--- Training LSTM Model ---")

MAX_NUM_WORDS = 5000
MAX_SEQUENCE_LENGTH = 100

# Tokenize text into sequences for deep learning
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(df["clean_text"])
sequences = tokenizer.texts_to_sequences(df["clean_text"])
X_lstm = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Train-Test Split for LSTM
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_lstm, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Convert labels to one-hot encoding for multi-class classification
num_classes = len(label_encoder.classes_)
y_train_cat = tf.keras.utils.to_categorical(y_train_lstm, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test_lstm, num_classes)

# Define LSTM Architecture
embedding_dim = 128

lstm_model = Sequential(
    [
        Embedding(
            input_dim=MAX_NUM_WORDS,
            output_dim=embedding_dim,
            input_length=MAX_SEQUENCE_LENGTH,
        ),
        SpatialDropout1D(0.2),
        LSTM(units=100, dropout=0.2, recurrent_dropout=0.2),
        Dense(units=num_classes, activation="softmax"),
    ]
)

lstm_model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Train the LSTM
history = lstm_model.fit(
    X_train_lstm,
    y_train_cat,
    epochs=5,
    batch_size=32,
    validation_data=(X_test_lstm, y_test_cat),
    verbose=1,
)

# Evaluate LSTM
lstm_preds = np.argmax(lstm_model.predict(X_test_lstm), axis=1)
print("\n[3] LSTM Accuracy:", accuracy_score(y_test_lstm, lstm_preds))
print(
    classification_report(
        y_test_lstm, lstm_preds, target_names=label_encoder.classes_
    )
)
