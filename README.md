# Twitter Sentiment Analysis using NLP and Machine Learning

A complete **Artificial Intelligence** project that performs **Twitter Sentiment Analysis** using **Natural Language Processing (NLP)** and **Machine Learning**. The model classifies tweets into **Positive**, **Negative**, and **Neutral** sentiments using text preprocessing, TF-IDF vectorization, and supervised learning algorithms.

---

## 📌 Project Overview

The objective of this project is to automatically analyze public opinions expressed on Twitter by classifying tweets into different sentiment categories. This project demonstrates the complete NLP pipeline, from text preprocessing to model training, evaluation, and prediction.

---

## 🎯 Objectives

- Perform sentiment analysis on Twitter data.
- Preprocess tweets using NLP techniques.
- Convert text into numerical features using TF-IDF.
- Train and evaluate machine learning models.
- Predict sentiments of new tweets.
- Build a foundation for real-time social media sentiment analysis.

---

## 📊 Dataset Information

- **Dataset Name:** Twitter Sentiment Dataset
- **Total Records:** **5,695 Tweets**
- **Classes:**
  - 😊 Positive
  - 😐 Neutral
  - 😠 Negative

### Dataset Features

| Column | Description |
|---------|-------------|
| `review` | Tweet text |
| `category` | Sentiment label |

### Sample Dataset

| Tweet | Sentiment |
|--------|-----------|
| I absolutely love this phone! | Positive |
| Worst experience ever. | Negative |
| The product is okay. | Neutral |

---

## 🛠 Technologies Used

- Python 3.11.9
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Matplotlib
- Joblib

---

## 📂 Project Structure

```text
twitter-sentiment-analysis/
│
├── dataset/
│   └── twitter_data.csv
│
├── app.py
├── sentiment_analysis.py
├── README.md
```

---

## ⚙️ Data Preprocessing

The following NLP preprocessing steps are applied:

- Convert text to lowercase
- Remove URLs
- Remove punctuation
- Remove numbers
- Remove special characters
- Tokenization
- Stopword removal
- Lemmatization

---

## 🔤 Feature Extraction

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert cleaned tweets into numerical feature vectors for machine learning.

---

## 🤖 Machine Learning Models

The following models are implemented:

- Logistic Regression
- Multinomial Naive Bayes
- Long Short-Term Memory (LSTM)

---

## 📈 Model Evaluation

Performance metrics include:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Expected Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 92–95% |
| Naive Bayes | 89–92% |
| LSTM | 94–97% |

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/twitter-sentiment-analysis.git
```

Move into the project folder:

```bash
cd twitter-sentiment-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Train the model:

```bash
python sentiment_analysis.py
```

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

---

## 📊 Output

The application predicts one of the following sentiments:

- 😊 Positive
- 😐 Neutral
- 😠 Negative

Example:

**Input**

```
This product is amazing. I highly recommend it.
```

**Prediction**

```
Positive 😊
Confidence: 97.8%
```

---

## 🌟 Applications

- Brand reputation monitoring
- Customer feedback analysis
- Product review classification
- Social media analytics
- Market research
- Business intelligence

---

## 🔮 Future Improvements

- Real-time Twitter API integration
- BERT/RoBERTa-based transformer models
- Multilingual sentiment analysis
- Interactive dashboards with advanced visualizations
- Cloud deployment using AWS, Azure, or Google Cloud

---

## 👨‍💻 Author

**Ravi Raj**

Artificial Intelligence Internship Project

---

## 📄 License

This project is intended for educational and internship purposes.
