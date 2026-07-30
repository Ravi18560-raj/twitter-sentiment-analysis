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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
class_names = label_encoder.classes_

# ==========================================
# PART 1: TF-IDF Models (Naive Bayes & Logistic Regression)
# ==========================================

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df["clean_text"])

# Train-Test Split for TF-IDF
X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(
    X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 1. Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tf, y_train_tf)
nb_preds = nb_model.predict(X_test_tf)
lr_acc = accuracy_score(y_test_tf, lr_preds)

# 2. Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tf, y_train_tf)
lr_preds = lr_model.predict(X_test_tf)
lr_acc = accuracy_score(y_test_tf, lr_preds)


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
lstm_model = Sequential(
    [
        Embedding(
            input_dim=MAX_NUM_WORDS,
            output_dim=128,
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
lstm_acc = accuracy_score(y_test_lstm, lstm_preds)

# ==========================================
# 4. Plot Visualizations
# ==========================================
sns.set_theme(style="whitegrid")

# --- CHART 1: Model Accuracy Comparison ---
plt.figure(figsize=(8, 5))
models = ["Naive Bayes", "Logistic Regression", "LSTM"]
accuracies = [nb_acc * 100, lr_acc * 100, lstm_acc * 100]

bars = plt.bar(models, accuracies, color=["#3498db", "#2ecc71", "#9b59b6"])
plt.title("Model Accuracy Comparison (%)", fontsize=14, fontweight="bold")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 1.5,
        f"{yval:.2f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()

# --- CHART 2: LSTM Training & Validation Curves (Accuracy & Loss) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# LSTM Accuracy Curve
ax1.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
ax1.plot(history.history["val_accuracy"], label="Val Accuracy", marker="o")
ax1.set_title("LSTM Accuracy Curve", fontweight="bold")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend()

# LSTM Loss Curve
ax2.plot(
    history.history["loss"], label="Train Loss", color="orange", marker="o"
)
ax2.plot(
    history.history["val_loss"], label="Val Loss", color="red", marker="o"
)
ax2.set_title("LSTM Loss Curve", fontweight="bold")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend()

plt.tight_layout()
plt.savefig("lstm_training_curves.png")
plt.show()

# --- CHART 3: Confusion Matrices for All 3 Models ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Naive Bayes Confusion Matrix
cm_nb = confusion_matrix(y_test_tf, nb_preds)
sns.heatmap(
    cm_nb,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axes[0],
)
axes[0].set_title("Naive Bayes Confusion Matrix")
axes[0].set_ylabel("Actual Label")
axes[0].set_xlabel("Predicted Label")

# Logistic Regression Confusion Matrix
cm_lr = confusion_matrix(y_test_tf, lr_preds)
sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axes[1],
)
axes[1].set_title("Logistic Regression Confusion Matrix")
axes[1].set_ylabel("Actual Label")
axes[1].set_xlabel("Predicted Label")

# LSTM Confusion Matrix
cm_lstm = confusion_matrix(y_test_lstm, lstm_preds)
sns.heatmap(
    cm_lstm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axes[2],
)
axes[2].set_title("LSTM Confusion Matrix")
axes[2].set_ylabel("Actual Label")
axes[2].set_xlabel("Predicted Label")

plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.show()

# Save Trained Models
import joblib

# 1. Save Label Encoder & Vectorizers
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(tokenizer, "tokenizer.pkl")

# 2. Save Classical ML Models
joblib.dump(nb_model, "naive_bayes.pkl")
joblib.dump(lr_model, "logistic_regression.pkl")

# 3. Save Keras LSTM Model
lstm_model.save("lstm_model.keras")

print("All models and vectorizers saved successfully!")
