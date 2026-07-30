import json
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    label_encoder = joblib.load("label_encoder.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    tokenizer = joblib.load("tokenizer.pkl")

    nb_model = joblib.load("naive_bayes.pkl")
    lr_model = joblib.load("logistic_regression.pkl")
    lstm_model = tf.keras.models.load_model("lstm_model.keras")

    return (
        label_encoder,
        tfidf,
        tokenizer,
        nb_model,
        lr_model,
        lstm_model,
    )


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
        df = pd.read_csv("cleaned_twitter_data.csv")
        counts = df["category"].value_counts()
        pcts = (df["category"].value_counts(normalize=True) * 100).round(2)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tweets", len(df))
        col2.metric("Positive Reviews", f"{pcts.get('Positive', 0)}%")
        col3.metric("Negative Reviews", f"{pcts.get('Negative', 0)}%")
        col4.metric("Neutral Reviews", f"{pcts.get('Neutral', 0)}%")

        st.markdown("---")

        c1, c2 = st.columns(2)
        color_map = {
            "Positive": "#2ecc71",
            "Negative": "#e74c3c",
            "Neutral": "#3498db",
        }

        with c1:
            st.subheader("Sentiment Percentage (Donut Chart)")
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
            df_bar = pd.DataFrame(
                {"Sentiment": pcts.index, "Percentage": pcts.values}
            )
            fig_bar = px.bar(
                df_bar,
                x="Sentiment",
                y="Percentage",
                color="Sentiment",
                color_discrete_map=color_map,
                text_auto=True,
            )
            fig_bar.update_layout(showlegend=False, yaxis_title="Percentage (%)")
            st.plotly_chart(fig_bar, use_container_width=True)

    except FileNotFoundError:
        st.error("Could not find `cleaned_twitter_data.csv`.")

# ----------------------------------------------------
# TAB 2: Model Performance
# ----------------------------------------------------
with tab2:
    st.header("Model Evaluation & Training Metrics")
    if os.path.exists("model_accuracies.json"):
        with open("model_accuracies.json") as f:
            acc_data = json.load(f)

        m1, m2, m3 = st.columns(3)
        m1.metric("Naive Bayes Accuracy", f"{acc_data.get('Naive Bayes')}%")
        m2.metric(
            "Logistic Regression Accuracy",
            f"{acc_data.get('Logistic Regression')}%",
        )
        m3.metric("LSTM Accuracy", f"{acc_data.get('LSTM')}%")
        st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("1. Accuracy Comparison Across Models")
        if os.path.exists("accuracy_comparison.png"):
            st.image("accuracy_comparison.png")

        st.subheader("2. LSTM Loss & Accuracy Curves")
        if os.path.exists("lstm_curves.png"):
            st.image("lstm_curves.png")

    with col_right:
        st.subheader("3. Confusion Matrices")
        if os.path.exists("confusion_matrices.png"):
            st.image("confusion_matrices.png")

# ----------------------------------------------------
# TAB 3: Real-Time Prediction Section
# ----------------------------------------------------
with tab3:
    st.header("🔮 Test Custom Tweets for Sentiment")
    st.write(
        "Enter any text below to see real-time predictions from all three models."
    )

    try:
        (
            label_encoder,
            tfidf,
            tokenizer,
            nb_model,
            lr_model,
            lstm_model,
        ) = load_artifacts()

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
            if user_input.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                # 1. Transform text for TF-IDF Models
                input_tfidf = tfidf.transform([user_input])

                # 2. Transform text for LSTM Model
                input_seq = tokenizer.texts_to_sequences([user_input])
                input_padded = pad_sequences(input_seq, maxlen=100)

                # Predictions
                nb_pred = label_encoder.inverse_transform(
                    nb_model.predict(input_tfidf)
                )[0]
                lr_pred = label_encoder.inverse_transform(
                    lr_model.predict(input_tfidf)
                )[0]

                lstm_probs = lstm_model.predict(input_padded)[0]
                lstm_pred_idx = np.argmax(lstm_probs)
                lstm_pred = label_encoder.inverse_transform([lstm_pred_idx])[0]

                # Map predictions dictionary
                model_results = {
                    "Naive Bayes": nb_pred,
                    "Logistic Regression": lr_pred,
                    "LSTM": lstm_pred,
                }

                chosen_result = model_results[selected_model]

                # Result Display Card
                st.markdown("### Primary Result")
                if chosen_result == "Positive":
                    st.success(
                        f"**Predicted Sentiment:** {chosen_result} 😊"
                    )
                elif chosen_result == "Negative":
                    st.error(f"**Predicted Sentiment:** {chosen_result} 😡")
                else:
                    st.info(f"**Predicted Sentiment:** {chosen_result} 😐")

                # Comparison Table across models
                st.markdown("---")
                st.subheader("Model Consensus Comparison")

                res_df = pd.DataFrame(
                    {
                        "Model": list(model_results.keys()),
                        "Predicted Sentiment": list(model_results.values()),
                    }
                )
                st.table(res_df)

                # LSTM Probabilities Bar Chart
                st.subheader("LSTM Prediction Confidence Breakdown")
                prob_df = pd.DataFrame(
                    {
                        "Sentiment": label_encoder.classes_,
                        "Probability (%)": (lstm_probs * 100).round(2),
                    }
                )
                fig_probs = px.bar(
                    prob_df,
                    x="Sentiment",
                    y="Probability (%)",
                    color="Sentiment",
                    text_auto=True,
                    color_discrete_map={
                        "Positive": "#2ecc71",
                        "Negative": "#e74c3c",
                        "Neutral": "#3498db",
                    },
                )
                fig_probs.update_layout(showlegend=False)
                st.plotly_chart(fig_probs, use_container_width=True)

    except FileNotFoundError:
        st.error(
            "Model files not found! Make sure you run `train_and_evaluate.py` to generate `.pkl` and `.keras` model files first."
        )
