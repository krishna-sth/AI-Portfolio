# app.py - Enhanced Streamlit Demo
import streamlit as st
import joblib
from spam_classifier import preprocess_text

# -------------------------------
# Load model and vectorizer once
# -------------------------------
model = joblib.load("model/spam_classifier_model_updated.pkl")
vectorizer = joblib.load("model/vectorizer_updated.pkl")

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Spam SMS Classifier", page_icon="📨")
st.title("📨 Spam SMS Classifier")
st.write("Enter a message below to check if it's **spam** or **ham**.")

msg = st.text_area("Message", height=100)

# -------------------------------
# Predict button
# -------------------------------
if st.button("Predict"):
    if msg.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        try:
            # Preprocess and transform the message
            msg_vector = vectorizer.transform([preprocess_text(msg)])
            
            # Predict label and probability
            pred_label = model.predict(msg_vector)[0]
            pred_prob = model.predict_proba(msg_vector)[0]
            spam_prob = pred_prob[1]  # Probability of being SPAM

            pred = 'SPAM' if pred_label == 1 else 'HAM'

            # Display prediction
            st.markdown(f"### Prediction: **{pred}**")
            
            # Show probability bar
            if pred == 'SPAM':
                st.warning(f"⚠️ This looks suspicious! Spam probability: {spam_prob*100:.2f}%")
                st.progress(int(spam_prob*100))
            else:
                st.success(f"✅ Looks safe! Ham probability: {(1-spam_prob)*100:.2f}%")
                st.progress(int((1-spam_prob)*100))
        except Exception as e:
            st.error(f"❌ Error occurred: {e}")
