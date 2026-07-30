import json
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

# ----------------------------------------------------
# Logging
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# Constants & File Paths
# ----------------------------------------------------
MAX_SEQ_LEN = 100
MODEL_FILES = {
    "label_encoder": "label_encoder.pkl",
    "tfidf": "tfidf_vectorizer.pkl",
    "tokenizer": "tokenizer.pkl",
    "nb": "naive_bayes.pkl",
    "lr": "logistic_regression.pkl",
    "lstm": "lstm_model.keras",
}
DATA_FILES = {
    "cleaned_data": "cleaned_twitter_data.csv",
    "accuracies": "model_accuracies.json",
    "accuracy_img": "accuracy_comparison.png",
    "lstm_curves": "lstm_curves.png",
    "confusion": "confusion_matrices.png",
}
DEFAULT_COLORS = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#3498db"}

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------

def assert_files_exist(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")


def decode_prediction(model_pred, label_encoder):
    """Decode different possible sklearn model outputs into label strings.
    Handles:
      - predict_proba / probability arrays (2D): take argmax
      - integer-encoded predictions: inverse_transform
      - string labels: return as-is
    Returns an array of label strings.
    """
    arr = np.asarray(model_pred)
    # Probabilities (n_samples, n_classes)
    if arr.ndim == 2 and arr.shape[1] > 1:
        idx = np.argmax(arr, axis=1)
        try:
            return label_encoder.inverse_transform(idx)
        except Exception:
            return idx.astype(str)
    # Single-dim probabilities for 1 sample but multi-class
    if arr.ndim == 1 and arr.size > 1 and not np.issubdtype(arr.dtype, np.integer):
        idx = np.argmax(arr, axis=0)
        try:
            return label_encoder.inverse_transform([idx])
        except Exception:
            return np.array([str(idx)])
    # Integer encoded labels
    if np.issubdtype(arr.dtype, np.integer):
        try:
            return label_encoder.inverse_transform(arr)
        except Exception:
            return arr.astype(str)
    # Otherwise assume already string labels
    return arr.astype(str)


def predict_all_models(text, tfidf, tokenizer, nb_model, lr_model, lstm_model, label_encoder, maxlen=MAX_SEQ_LEN):
    """Pure function: given text and artifacts, return predicted labels and LSTM probs.
    Returns:
      model_results: dict mapping model name -> label string
      lstm_probs: 1D numpy array of probabilities matching label_encoder.classes_
    Throws descriptive exceptions on mismatch or unexpected shapes.
    """
    if not isinstance(text, str) or text.strip() == "":
        raise ValueError("Input text must be a non-empty string")
    text = text.strip()

    # TF-IDF transform
    x_tfidf = tfidf.transform([text])

    # LSTM sequence
    x_seq = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=maxlen)

    # Raw predictions
    nb_raw = nb_model.predict(x_tfidf)
    lr_raw = lr_model.predict(x_tfidf)
    lstm_raw = lstm_model.predict(x_seq)

    # Decode NB and LR
    nb_label = decode_prediction(nb_raw, label_encoder)[0]
    lr_label = decode_prediction(lr_raw, label_encoder)[0]

    # Normalize LSTM output to 1D prob vector
    lstm_arr = np.asarray(lstm_raw)
    if lstm_arr.ndim == 2:
        probs = lstm_arr[0]
    elif lstm_arr.ndim == 1:
        probs = lstm_arr
    else:
        raise ValueError(f"Unexpected LSTM output shape: {lstm_arr.shape}")

    if probs.shape[0] != len(label_encoder.classes_):
        raise ValueError(
            "LSTM output length does not match number of classes in label encoder"
        )
    lstm_idx = int(np.argmax(probs))
    try:
        lstm_label = label_encoder.inverse_transform([lstm_idx])[0]
    except Exception as e:
        # Fallback to string index if inverse_transform fails
        logger.warning("label_encoder.inverse_transform failed for LSTM: %s", e)
        lstm_label = str(lstm_idx)

    return {"Naive Bayes": nb_label, "Logistic Regression": lr_label, "LSTM": lstm_label}, probs


# ----------------------------------------------------
# Page Config & Caching Models
# ----------------------------------------------------
st.set_page_config(
    page_title="Twitter Sentiment & Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    # Validate files first
    try:
        assert_files_exist(list(MODEL_FILES.values()))
    except FileNotFoundError as e:
        logger.error("Artifact files missing: %s", e)
        raise

    try:
        label_encoder = joblib.load(MODEL_FILES["label_encoder"])
    except Exception as e:
        logger.exception("Failed to load label encoder: %s", e)
        raise

    try:
        tfidf = joblib.load(MODEL_FILES["tfidf"])
    except Exception as e:
        logger.exception("Failed to load tfidf vectorizer: %s", e)
        raise

    try:
        tokenizer = joblib.load(MODEL_FILES["tokenizer"])
    except Exception as e:
        logger.exception("Failed to load tokenizer: %s", e)
        raise

    try:
        nb_model = joblib.load(MODEL_FILES["nb"])
    except Exception as e:
        logger.exception("Failed to load Naive Bayes model: %s", e)
        raise

    try:
        lr_model = joblib.load(MODEL_FILES["lr"])
    except Exception as e:
        logger.exception("Failed to load Logistic Regression model: %s", e)
        raise

    try:
        lstm_model = tf.keras.models.load_model(MODEL_FILES["lstm"])
    except Exception as e:
        logger.exception("Failed to load LSTM model: %s", e)
        raise

    return label_encoder, tfidf, tokenizer, nb_model, lr_model, lstm_model


st.title("📊 Twitter Sentiment Analysis Dashboard")

tab1, tab2, tab3 = st.tabs(
    [
        "📈 Sentiment Distribution",
        "🧠 Model Performance",
        "🔮 Live Tweet Prediction",
    ]
)

# ----------------------------------------------------
# TAB 1: Sentiment Distribution
# ----------------------------------------------------
with tab1:
    st.header("Dataset Overview & Sentiment Percentages")
    try:
        try:
            df = pd.read_csv(DATA_FILES["cleaned_data"])
        except FileNotFoundError:
            st.error(f"Could not find `{DATA_FILES['cleaned_data']}`.")
            df = None
        except pd.errors.EmptyDataError:
            st.error("Dataset file is empty")
            df = None
        except pd.errors.ParserError as e:
            st.error(f"Error parsing CSV: {e}")
            df = None

        if df is not None:
            counts = df["category"].value_counts()
            pcts = (df["category"].value_counts(normalize=True) * 100).round(2)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Tweets", len(df))
            col2.metric("Positive Reviews", f"{pcts.get('Positive', 0)}%")
            col3.metric("Negative Reviews", f"{pcts.get('Negative', 0)}%")
            col4.metric("Neutral Reviews", f"{pcts.get('Neutral', 0)}%")

            st.markdown("---")

            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Sentiment Percentage (Donut Chart)")
                # build color map defensively
                color_map = {k: DEFAULT_COLORS.get(k, "#95a5a6") for k in counts.index}
                fig_pie = px.pie(
                    values=counts.values,
                    names=counts.index,
                    hole=0.4,
                    color=counts.index,
                    color_discrete_map=color_map,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                st.subheader("Sentiment Bar Breakdown")
                df_bar = pd.DataFrame({"Sentiment": pcts.index, "Percentage": pcts.values})
                fig_bar = px.bar(
                    df_bar,
                    x="Sentiment",
                    y="Percentage",
                    color="Sentiment",
                    color_discrete_map={k: DEFAULT_COLORS.get(k, "#95a5a6") for k in pcts.index},
                    text_auto=True,
                )
                fig_bar.update_layout(showlegend=False, yaxis_title="Percentage (%)")
                st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        logger.exception("Unexpected error in Tab 1: %s", e)
        st.error("An unexpected error occurred while loading the dataset.")

# ----------------------------------------------------
# TAB 2: Model Performance
# ----------------------------------------------------
with tab2:
    st.header("Model Evaluation & Training Metrics")
    if os.path.exists(DATA_FILES["accuracies"]):
        try:
            with open(DATA_FILES["accuracies"]) as f:
                acc_data = json.load(f)

            m1, m2, m3 = st.columns(3)
            m1.metric("Naive Bayes Accuracy", f"{acc_data.get('Naive Bayes')}%")
            m2.metric("Logistic Regression Accuracy", f"{acc_data.get('Logistic Regression')}%")
            m3.metric("LSTM Accuracy", f"{acc_data.get('LSTM')}%")
            st.markdown("---")
        except json.JSONDecodeError:
            st.error("model_accuracies.json is not valid JSON")
        except Exception as e:
            logger.exception("Error reading accuracies file: %s", e)
            st.error("Could not read model_accuracies.json")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("1. Accuracy Comparison Across Models")
        if os.path.exists(DATA_FILES["accuracy_img"]):
            st.image(DATA_FILES["accuracy_img"])

        st.subheader("2. LSTM Loss & Accuracy Curves")
        if os.path.exists(DATA_FILES["lstm_curves"]):
            st.image(DATA_FILES["lstm_curves"])

    with col_right:
        st.subheader("3. Confusion Matrices")
        if os.path.exists(DATA_FILES["confusion"]):
            st.image(DATA_FILES["confusion"])

# ----------------------------------------------------
# TAB 3: Real-Time Prediction Section
# ----------------------------------------------------
with tab3:
    st.header("🔮 Test Custom Tweets for Sentiment")
    st.write("Enter any text below to see real-time predictions from all three models.")

    try:
        label_encoder, tfidf, tokenizer, nb_model, lr_model, lstm_model = load_artifacts()

        # Input box
        user_input = st.text_area(
            "Type a tweet or review:",
            value="I really love using this product, it works amazingly well!",
            height=100,
        )

        selected_model = st.selectbox(
            "Choose Model for Primary Prediction:",
            ["Logistic Regression", "Naive Bayes", "LSTM"],
        )

        if st.button("Analyze Sentiment", type="primary"):
            user_input = (user_input or "").strip()
            if user_input == "":
                st.warning("Please enter some text to analyze.")
            elif len(user_input) > 5000:
                st.warning("Input too long; please shorten the text to under 5000 characters.")
            else:
                try:
                    model_results, lstm_probs = predict_all_models(
                        user_input,
                        tfidf,
                        tokenizer,
                        nb_model,
                        lr_model,
                        lstm_model,
                        label_encoder,
                        maxlen=MAX_SEQ_LEN,
                    )

                    chosen_result = model_results[selected_model]

                    # Result Display Card
                    st.markdown("### Primary Result")
                    if chosen_result == "Positive":
                        st.success(f"**Predicted Sentiment:** {chosen_result} 😊")
                    elif chosen_result == "Negative":
                        st.error(f"**Predicted Sentiment:** {chosen_result} 😡")
                    else:
                        st.info(f"**Predicted Sentiment:** {chosen_result} 😐")

                    # Comparison Table across models
                    st.markdown("---")
                    st.subheader("Model Consensus Comparison")

                    res_df = pd.DataFrame({
                        "Model": list(model_results.keys()),
                        "Predicted Sentiment": list(model_results.values()),
                    })
                    st.table(res_df)

                    # LSTM Probabilities Bar Chart
                    st.subheader("LSTM Prediction Confidence Breakdown")
                    prob_df = pd.DataFrame({
                        "Sentiment": label_encoder.classes_,
                        "Probability (%)": (lstm_probs * 100).round(2),
                    })
                    fig_probs = px.bar(
                        prob_df,
                        x="Sentiment",
                        y="Probability (%)",
                        color="Sentiment",
                        text_auto=True,
                        color_discrete_map={k: DEFAULT_COLORS.get(k, "#95a5a6") for k in label_encoder.classes_},
                    )
                    fig_probs.update_layout(showlegend=False)
                    st.plotly_chart(fig_probs, use_container_width=True)

                except Exception as e:
                    logger.exception("Prediction failed: %s", e)
                    st.error(f"Prediction failed: {e}")

    except FileNotFoundError:
        st.error(
            "Model files not found! Make sure you run `train_and_evaluate.py` to generate `.pkl` and `.keras` model files first."
        )
    except Exception as e:
        logger.exception("Failed to load artifacts: %s", e)
        st.error("Failed to load model artifacts. See logs for details.")
