import streamlit as st
import joblib

# Title
st.title("Spam Detection with best_mnb Model")
st.write("Enter a message to classify whether it's Spam or Not Spam.")

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load('best_mnb.pkl')
    return model

model = load_model()

# User Input
user_input = st.text_area("Enter your message here", "")

# Prediction
if st.button("Classify"):
    if user_input:
        prediction = model.predict([user_input])[0]
        
        if prediction == 1:
            st.success("### Prediction: 🛑 **1 (Spam)**")
        else:
            st.success("### Prediction: ✅ **0 (Not Spam)**")
    else:
        st.warning("Please enter a message to classify.")
