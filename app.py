import streamlit as st
import pickle
import re
import numpy as np

# ===============================
# PAGE CONFIG (MUST BE FIRST)
# ===============================
st.set_page_config(
    page_title="Fake Review Detection System",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ===============================
# LOAD MODEL & VECTORIZER
# ===============================
model = pickle.load(open("model/fake_review_model.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

# ===============================
# TEXT CLEANING FUNCTION
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ===============================
# SIDEBAR (PROFESSIONAL)
# ===============================
with st.sidebar:
    st.title("📌 Project Info")
    st.markdown("""
    **Fake Review Detection System**

    This application uses **NLP & Machine Learning** to analyze
    e-commerce product reviews and classify them as:

    - ✅ Genuine Review  
    - ❌ Fake Review  

    **Tech Used**
    - TF-IDF
    - Logistic Regression
    - Streamlit
    """)

    st.markdown("---")
    st.info("👩‍💻 Developed by **Nidhi Ghatole**")

# ===============================
# MAIN PAGE UI
# ===============================
st.markdown(
    "<h1 style='text-align: center;'>🛒 Fake Review Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Detect whether an e-commerce product review is <b>Fake</b> or <b>Genuine</b> using AI"
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ===============================
# INPUT SECTION
# ===============================
st.subheader("✍️ Enter Product Review")

review = st.text_area(
    "Paste the product review below:",
    height=180,
    placeholder="Example: I have been using this product for 6 months and it works perfectly..."
)

# ===============================
# PREDICTION BUTTON
# ===============================
if st.button("🔍 Analyze Review", use_container_width=True):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review before clicking Analyze.")
    else:
        cleaned_review = clean_text(review)
        vectorized_review = tfidf.transform([cleaned_review])

        prediction = model.predict(vectorized_review)[0]
        confidence = np.max(model.predict_proba(vectorized_review))

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success("✅ **Genuine Review**")
            st.progress(int(confidence * 100))
            st.write(f"**Confidence:** {confidence:.2f}")
        else:
            st.error("❌ **Fake Review**")
            st.progress(int(confidence * 100))
            st.write(f"**Confidence:** {confidence:.2f}")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 13px; color: gray;'>"
    "This project is for educational and research purposes only.<br>"
    "Built using NLP & Machine Learning."
    "</p>",
    unsafe_allow_html=True
)