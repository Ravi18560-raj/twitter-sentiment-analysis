"""
Patched sentiment_analysis.py
- Consolidated imports and removed duplicates
- Robust NLTK resource checks/downloads
- Fixed variable bugs (nb_acc/lr_acc order)
- Consolidated TF-IDF creation and use
- Wrapped logic in functions and `if __name__ == '__main__'` guard
- Better error handling and informative prints
"""

import os
import re
import sys
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

import joblib
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure NLTK resources are available (download if missing)
def ensure_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        logging.info("Downloading punkt tokenizer...")
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        logging.info("Downloading stopwords...")
        nltk.download("stopwords", quiet=True)


def preprocess_text(text: str) -> str:
    """Lowercase, remove URLs/mentions/hashtags/numbers/punctuation, tokenize, and remove stopwords."""
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs, mentions, hashtags, punctuation and numbers
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()
    filtered_tokens = [w for w in tokens if w not in stop_words]

    return " ".join(filtered_tokens)


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        logging.error("Input CSV not found: %s", csv_path)
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "tweet" not in df.columns:
        logging.error("Expected column 'tweet' not found in %s", csv_path)
        raise KeyError("Expected column 'tweet' not found in input CSV")

    logging.info("Preprocessing %d rows from %s", len(df), csv_path)
    df["clean_text"] = df["tweet"].apply(preprocess_text)
    df["clean_text"] = df["clean_text"].fillna("")

    # Ensure 'category' exists
    if "category" not in df.columns:
        logging.error("Expected column 'category' not found in %s", csv_path)
        raise KeyError("Expected column 'category' not found in input CSV")

    return df[["clean_text", "category"]]


def train_and_evaluate(df: pd.DataFrame, random_state: int = 42):
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df["category"])
    class_names = label_encoder.classes_

    # TF-IDF vectorizer (use unigrams + bigrams)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df["clean_text"])

    # Train-Test split for classical models
    X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(
        X_tfidf, y_encoded, test_size=0.2, random_state=random_state, stratify=y_encoded
    )

    # 1) Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tf, y_train_tf)
    nb_preds = nb_model.predict(X_test_tf)
    nb_acc = accuracy_score(y_test_tf, nb_preds)
    logging.info("Naive Bayes accuracy: %.4f", nb_acc)

    # 2) Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tf, y_train_tf)
    lr_preds = lr_model.predict(X_test_tf)
    lr_acc = accuracy_score(y_test_tf, lr_preds)
    logging.info("Logistic Regression accuracy: %.4f", lr_acc)

    # Prepare data for LSTM (tokenizer should be fit on whole dataset)
    MAX_NUM_WORDS = 5000
    MAX_SEQUENCE_LENGTH = 100

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token=None)
    tokenizer.fit_on_texts(df["clean_text"])
    sequences = tokenizer.texts_to_sequences(df["clean_text"])
    X_lstm = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
        X_lstm, y_encoded, test_size=0.2, random_state=random_state, stratify=y_encoded
    )

    num_classes = len(class_names)
    y_train_cat = tf.keras.utils.to_categorical(y_train_lstm, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test_lstm, num_classes)

    # Define LSTM model
    lstm_model = Sequential(
        [
            Embedding(input_dim=MAX_NUM_WORDS, output_dim=128, input_length=MAX_SEQUENCE_LENGTH),
            SpatialDropout1D(0.2),
            LSTM(units=100, dropout=0.2, recurrent_dropout=0.2),
            Dense(units=num_classes, activation="softmax"),
        ]
    )

    lstm_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train LSTM (keep epochs small by default; caller may increase)
    logging.info("Training LSTM model...")
    history = lstm_model.fit(
        X_train_lstm,
        y_train_cat,
        epochs=5,
        batch_size=32,
        validation_data=(X_test_lstm, y_test_cat),
        verbose=1,
    )

    # Evaluate LSTM
    lstm_probs = lstm_model.predict(X_test_lstm)
    lstm_preds = np.argmax(lstm_probs, axis=1)
    lstm_acc = accuracy_score(y_test_lstm, lstm_preds)
    logging.info("LSTM accuracy: %.4f", lstm_acc)

    # Plot results
    plot_results(nb_acc, lr_acc, lstm_acc, history, class_names, y_test_tf, nb_preds, lr_preds, y_test_lstm, lstm_preds)

    # Save models and artifacts
    joblib.dump(label_encoder, "label_encoder.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    joblib.dump(tokenizer, "tokenizer.pkl")

    joblib.dump(nb_model, "naive_bayes.pkl")
    joblib.dump(lr_model, "logistic_regression.pkl")

    # Save Keras model
    lstm_model.save("lstm_model.keras")

    logging.info("All models and vectorizers saved successfully!")

    return {
        "nb_acc": nb_acc,
        "lr_acc": lr_acc,
        "lstm_acc": lstm_acc,
        "label_encoder": label_encoder,
        "tfidf": tfidf,
        "tokenizer": tokenizer,
    }


def plot_results(nb_acc, lr_acc, lstm_acc, history, class_names, y_test_tf, nb_preds, lr_preds, y_test_lstm, lstm_preds):
    sns.set_theme(style="whitegrid")

    # Chart 1: Model Accuracy Comparison
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
    plt.close()

    # Chart 2: LSTM Training & Validation Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history.get("accuracy", []), label="Train Accuracy", marker="o")
    ax1.plot(history.history.get("val_accuracy", []), label="Val Accuracy", marker="o")
    ax1.set_title("LSTM Accuracy Curve", fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(history.history.get("loss", []), label="Train Loss", color="orange", marker="o")
    ax2.plot(history.history.get("val_loss", []), label="Val Loss", color="red", marker="o")
    ax2.set_title("LSTM Loss Curve", fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("lstm_training_curves.png")
    plt.close()

    # Chart 3: Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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
    plt.close()


if __name__ == "__main__":
    ensure_nltk_resources()

    # The script expects the raw twitter_data.csv to exist in the working directory.
    INPUT_CSV = "twitter_data.csv"

    try:
        cleaned_df = load_and_preprocess(INPUT_CSV)
    except Exception as e:
        logging.error("Failed to load/preprocess data: %s", e)
        sys.exit(1)

    # Save cleaned data
    cleaned_df.to_csv("cleaned_twitter_data.csv", index=False)
    logging.info("Saved cleaned data to cleaned_twitter_data.csv")

    results = train_and_evaluate(cleaned_df)
    logging.info("Done. Results summary: %s", {k: v for k, v in results.items() if isinstance(v, (int, float, str))})
