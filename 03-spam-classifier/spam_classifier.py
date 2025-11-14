
# spam_classifier.py
import pandas as pd
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Download NLTK data (quiet)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text: str) -> str:
    """Clean and lemmatize text."""
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def train_and_save():
    """Train model and save."""
    df = pd.read_csv("data/SMS_spam_collection.csv")
    df['cleaned'] = df['Message'].apply(preprocess_text)
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['Label'].map({'ham': 0, 'spam': 1})
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    joblib.dump(model, "model/spam_classifier_model_updated.pkl")
    joblib.dump(vectorizer, "model/vectorizer_updated.pkl")
    print("Model and vectorizer saved to 'model/'")

def predict_message(msg: str) -> str:
    """Predict if message is spam."""
    cleaned = preprocess_text(msg)
    vect_msg = joblib.load("model/vectorizer_updated.pkl").transform([cleaned])
    pred = joblib.load("model/spam_classifier_model_updated.pkl").predict(vect_msg)
    return 'spam' if pred[0] == 1 else 'ham'

if __name__ == "__main__":
    train_and_save()
    print("\nTest:", predict_message("Win $1000! Click here"))
