import time
import streamlit as st
import pickle
from preprocess import clean_text
from bert_model import predict_bert
from api import get_news




def animated_warning():
    text = "⚠️ This model detects fake news based on language patterns, not real-world facts."
    placeholder = st.empty()
    typed = ""

    for char in text:
        typed += char
        placeholder.markdown(f"""
        <div style='background:#1c2b3a; padding:12px; border-radius:10px; color:#9ecfff; font-size:15px;'>
        {typed}<span style='color:#00ffcc;'>|</span>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(0.03)


st.set_page_config(page_title="Fake News Detector", layout="wide")

# =========================
#  CSS
# =========================
st.markdown("""
<style>
.main {background-color: #0E1117;}

/* CARD ANIMATION */
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    animation: slideUp 0.8s ease;
}

@keyframes slideUp {
    from {opacity: 0; transform: translateY(40px);}
    to {opacity: 1; transform: translateY(0);}
}

/* BUTTON EFFECT */
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    transition: all 0.2s ease;
}

.stButton>button:hover {
    transform: scale(1.08);
}

/* RESULT GLOW */
.glow {
    animation: glowPulse 1.5s infinite;
}

@keyframes glowPulse {
    0% {text-shadow: 0 0 5px #4CAF50;}
    50% {text-shadow: 0 0 20px #4CAF50;}
    100% {text-shadow: 0 0 5px #4CAF50;}
}

/* TEXTAREA */
textarea {
    background-color: #262730 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# =========================
#  LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

# =========================
#  SIDEBAR
# =========================
st.sidebar.title("📊 Model Details")
st.sidebar.markdown("""
### 🤖 Models Used
- Voting Classifier (ML)
- BERT (Deep Learning)

### 📈 Accuracy
~92–95% (Hybrid)

### 🔬 Improvement
- Hybrid ML + BERT decision

### ⚠️ Limitation
- Based on patterns, not real facts
""")

# =========================
#  ML PREDICT FUNCTION
# =========================
def predict(text):
    text = clean_text(text)
    vector = tfidf.transform([text])

    prediction = model.predict(vector)
    proba = model.predict_proba(vector)

    confidence = max(proba[0]) * 100
    label = "Fake ❌" if prediction[0] == 1 else "Real ✅"

    return label, round(confidence, 2)



# =========================
#  UI START
# =========================
# Title Animation
if "title_done" not in st.session_state:

    title = "🧠 Live Fake News Detector"
    title_placeholder = st.empty()

    for i in range(len(title) + 1):
        title_placeholder.markdown(
            f"<h1 style='text-align: center; color: #4CAF50;'>{title[:i]}</h1>",
            unsafe_allow_html=True
        )
        time.sleep(0.05)

    st.session_state.title_done = True

else:
    st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>
    🧠 Live Fake News Detector
    </h1>
    """, unsafe_allow_html=True)


# Warning Animation
if "warning_done" not in st.session_state:
    animated_warning()
    st.session_state.warning_done = True
else:
    st.markdown("""
    <div style='background:#1c2b3a; padding:12px; border-radius:10px; color:#9ecfff;'>
    ⚠️ This model detects fake news based on language patterns, not real-world facts.
    </div>
    """, unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Predict News", "🌐 Live News"])

# =========================
#  TAB 1: PREDICT
# =========================
with tab1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("✍️ Check Your Own News")
    user_input = st.text_area("Enter News Headline or Article")

    if st.button("🚀 Predict News"):
        fake_keywords = [
            "invisible", "time travel", "100% cure",
            "mind reading", "magic", "secret trick"
        ]

        keyword_flag = any(word in user_input.lower() for word in fake_keywords)

        if keyword_flag:
            st.warning("⚠️ Suspicious / unrealistic claim detected")

        if user_input.strip() != "":

            with st.spinner("🤖 AI is analyzing the news..."):
                time.sleep(1.2)

                label1, conf1 = predict(user_input)
                label2, conf2 = predict_bert(user_input)

            # =========================
            #  METRICS
            # =========================
            col1, col2, col3 = st.columns(3)

            col1.metric("ML Confidence", f"{conf1}%")
            col2.metric("BERT Confidence", f"{conf2}%")

            # =========================
            #  HYBRID DECISION
            # =========================
            if keyword_flag:
                final_label = "Fake ❌"
                final_conf = 80

            elif "Fake" in label1 and conf1 > 55:
                final_label = "Fake ❌"
                final_conf = conf1

            elif "Fake" in label2 and conf2 > 70:
                final_label = "Fake ❌"
                final_conf = conf2

            else:
                final_label = "Real ✅"
                final_conf = max(conf1, conf2)

            col3.metric("Final Result", final_label)

            # =========================
            #  GRAPH
            # =========================
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(4, 3))

            models = ["ML", "BERT"]
            confidences = [conf1, conf2]

            ax.bar(models, confidences)

            ax.set_ylim(0, 110)
            ax.set_ylabel("Confidence (%)")
            ax.set_title("Model Confidence Comparison")

            for i, v in enumerate(confidences):
                ax.text(i, v + 1, f"{round(v,1)}%", ha='center')

            st.pyplot(fig)

            # =========================
            #  FINAL RESULT DISPLAY
            # =========================
            st.markdown("### 🔍 Prediction Result")

            with st.spinner("🤖 Finalizing result..."):
                time.sleep(1)

            if "Fake" in final_label:
                st.markdown("""
                <h1 class='glow' style='text-align:center; color:red;'>
                🚨 FAKE NEWS
                </h1>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <h1 class='glow' style='text-align:center; color:#4CAF50;'>
                ✅ REAL NEWS
                </h1>
                """, unsafe_allow_html=True)

            progress = st.progress(0)

            for i in range(0, int(final_conf), 2):
                progress.progress(i)
                time.sleep(0.02)

            progress.progress(int(final_conf))
                
            

            st.write(f"Final Confidence: {round(final_conf,2)}%")
            st.caption("🔬 Result based on Hybrid ML + BERT Model")
            with st.expander("🧠 AI Thinking Process"):

                st.markdown("### 🔍 Step 1: Text Analysis")
                st.write("Cleaning and analyzing text...")
                time.sleep(0.5)

                st.markdown("### 🤖 Step 2: ML Model")
                st.success(f"{label1} ({round(conf1,2)}%)")
                time.sleep(0.5)

                st.markdown("### 🧠 Step 3: BERT Model")
                st.info(f"{label2} ({round(conf2,2)}%)")
                time.sleep(0.5)

                st.markdown("### ⚙️ Step 4: Decision Logic")
                st.warning("Applying hybrid decision logic...")
                time.sleep(0.5)

                st.markdown("### 🎯 Final Decision")
                if "Fake" in final_label:
                   st.error(final_label)
                else:
                     st.success(final_label)

        else:
            st.warning("Please enter some text")

    st.markdown('</div>', unsafe_allow_html=True)
    
    def type_text(text):
        placeholder = st.empty()
        typed = ""

        for char in text:
            typed += char
            placeholder.markdown(typed)
            time.sleep(0.02)

    type_text("🔍 Generating explanation using AI models...")

# =========================
#  TAB 2: LIVE NEWS
# =========================
with tab2:

    if st.button("📡 Fetch Live News"):

        with st.spinner("📡 Loading news..."):

            articles = get_news()
            cols = st.columns(2)

        for i, article in enumerate(articles):

            with cols[i % 2]:

                st.markdown("---")
                st.subheader(article['title'])

                if article['content']:

                   # 🔍 Prediction
                   label1, conf1 = predict(article['content'])
                   label2, conf2 = predict_bert(article['content'])

                   # 🧠 Hybrid Logic
                   if "Fake" in label1 and conf1 > 55:
                       final_label = "Fake ❌"
                       final_conf = conf1

                   elif "Fake" in label2 and conf2 > 70:
                       final_label = "Fake ❌"
                       final_conf = conf2

                   else:
                       final_label = "Real ✅"
                       final_conf = max(conf1, conf2)

                   # 🎨 Result UI
                   if "Fake" in final_label:
                       st.markdown("""<div style='background:#2b0f0f; padding:10px; border-radius:10px; border-left:5px solid red;'>
                       <b style='color:red;'>🚨 Fake News Detected</b></div>""", unsafe_allow_html=True)
                   else:
                       st.markdown("""<div style='background:#0f2b1b; padding:10px; border-radius:10px; border-left:5px solid #4CAF50;'>
                       <b style='color:#4CAF50;'>✅ Real News</b></div>""", unsafe_allow_html=True)

                       st.progress(int(final_conf))
                       st.write(f"Confidence: {round(final_conf,2)}%")

                else:
                    st.write("No content available")        

# =========================
#  FOOTER
# =========================
st.markdown("---")
st.markdown("👨‍💻 Developed by B12 | 🧠 AI Fake News Detection System")