# Spam SMS Classifier (97% F1)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-orange)

A **real-time spam detector** for SMS messages using **Multinomial Naive Bayes** and **TF-IDF vectorization**.  
Achieves **97% accuracy** and **90% spam recall**.

Live Demo: [Try it now!](https://your-streamlit-link.streamlit.app) *(coming soon)*

---

## Features

- **Preprocessing**: Lowercase, remove punctuation, stopwords, lemmatization
- **Model**: Multinomial Naive Bayes
- **Vectorization**: CountVectorizer (TF-IDF-like)
- **Performance**:
precision    recall  f1-score   support
ham       0.99      0.98      0.98      1448
spam       0.88      0.92      0.90       224
accuracy                           0.97      1672


---

## Project Structure
```text
03-spam-classifier/
├── spam_classifier.py          # Training + prediction logic
├── app.py                      # Streamlit live demo
├── explore_training.ipynb      # Optional: data exploration
├── requirements.txt
├── data/
│   └── SMS_spam_collection.csv
├── model/
│   ├── spam_classifier_model_updated.pkl
│   └── vectorizer_updated.pkl
└── README.md
```

---

## Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train & save model
python spam_classifier.py

# 3. Launch web demo
streamlit run app.py
```

## Quick Prediction

```python
from spam_classifier import predict_message

print(predict_message("Win $1000! Click here"))        # → 'spam'
print(predict_message("Hey, are you free tonight?"))  # → 'ham'
```

---

## Dataset
- **Source**: SMS Spam Collection
- **Size**: 5,572 messages (87% ham, 13% spam)
- **Format**: CSV with label and message columns

---

