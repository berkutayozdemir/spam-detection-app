import streamlit as st
import joblib

# Başlık
st.title("📧 Spam Tespit Uygulaması")
st.write("ingilizce Bir mesaj girin ve spam olup olmadığını öğrenin.")

# Model ve TF-IDF Vektörleştirici Yükleme
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load('best_mnb_model.pkl')  # Kaydedilen model dosyası
    vectorizer = joblib.load('vectorizer.pkl')  # Kaydedilen TF-IDF vektörleştirici
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Kullanıcı Girişi
user_input = st.text_area("📝 Mesajınızı buraya yazın:", "")

# Tahmin
if st.button("🧠 Sınıflandır"):
    if user_input.strip():
        # Metni TF-IDF vektörleştirici ile dönüştür
        user_input_vectorized = vectorizer.transform([user_input])
        
        # Tahmin yap
        prediction = model.predict(user_input_vectorized)[0]
        
        if prediction == 1:
            st.error("🛑 **Tahmin: Bu bir Spam mesajdır! (1)**")
        else:
            st.success("✅ **Tahmin: Bu bir Spam değil. (0)**")
    else:
        st.warning("⚠️ Lütfen sınıflandırmak için bir mesaj girin.")

