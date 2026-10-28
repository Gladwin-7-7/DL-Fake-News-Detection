import streamlit as st
import joblib
from datetime import datetime

st.set_page_config(
    page_title="TruthLens AI | AD3501",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- PROFESSIONAL DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #111a2e;
    border-right: 1px solid #26344d;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(120deg, #172554, #164e63);
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #315579;
    margin-bottom: 24px;
}
.hero h1 {
    color: #ffffff;
    font-size: 34px;
    margin-bottom: 8px;
}
.hero p {
    color: #cbd5e1;
    font-size: 16px;
}
.panel {
    background: #111a2e;
    padding: 22px;
    border: 1px solid #26344d;
    border-radius: 16px;
    margin-bottom: 16px;
}
.small-label {
    color: #94a3b8;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
div.stButton > button {
    background: #2563eb;
    color: white;
    border: 0;
    border-radius: 9px;
    min-height: 44px;
    font-weight: 600;
}
div.stButton > button:hover {
    background: #1d4ed8;
    color: white;
    border: 0;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return joblib.load("fake_news_model.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(
        "The model could not be loaded. Check that "
        "fake_news_model.joblib is in the project folder."
    )
    st.stop()

# ---------- SESSION DATA ----------
if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🔎 TruthLens AI")
    st.caption("AD3501 · Mini Project")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["News Analyzer", "Prediction History", "About Project"]
    )

    st.divider()
    st.markdown("### Model Information")
    st.write("*Algorithm*")
    st.caption("Logistic Regression")
    st.write("*Text processing*")
    st.caption("TF-IDF")
    st.write("*Framework*")
    st.caption("Python · Streamlit")

    st.divider()
    st.caption("AI predictions can be incorrect.")

# ---------- NEWS ANALYZER ----------
if page == "News Analyzer":

    st.markdown("""
    <div class="hero">
        <p class="small-label">AI-POWERED TEXT ANALYSIS</p>
        <h1>News credibility analyzer</h1>
        <p>Analyze news text using machine learning.
        Review the model's prediction before drawing conclusions.</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.8, 1], gap="large")

    with left:
        st.markdown("### 📝 Enter news content")

        examples = {
            "Choose an example": "",
            "Research headline": (
                "Scientists announce a new breakthrough in "
                "renewable energy research."
            ),
            "Sensational claim": (
                "BREAKING: Everyone will receive free money "
                "tomorrow! Share this secret message immediately."
            )
        }

        choice = st.selectbox(
            "Try a sample",
            list(examples.keys())
        )

        if st.button("Use selected example"):
            st.session_state.news_input = examples[choice]

        with st.form("news_form"):
            news = st.text_area(
                "Headline or article",
                value=st.session_state.get("news_input", ""),
                height=230,
                placeholder="Paste a news headline or full article here..."
            )

            submitted = st.form_submit_button(
                "🔍  Analyze News",
                use_container_width=True
            )

        if submitted:
            if len(news.strip()) < 20:
                st.warning("Please enter at least 20 characters.")
            else:
                prediction = model.predict([news])[0]
                probabilities = model.predict_proba([news])[0]
                classes = model.named_steps["classifier"].classes_

                scores = {
                    label: float(prob)
                    for label, prob in zip(classes, probabilities)
                }

                st.session_state.result = {
                    "prediction": prediction,
                    "confidence": max(scores.values()) * 100,
                    "scores": scores,
                    "text": news
                }

                st.session_state.history.insert(0, {
                    "Time": datetime.now().strftime("%d-%m-%Y %H:%M"),
                    "Prediction": prediction,
                    "News": news[:100]
                })

    with right:
        st.markdown("""
        <div class="panel">
            <p class="small-label">HOW IT WORKS</p>
            <h3>01 · Enter news</h3>
            <p>Paste a headline or article.</p>
            <h3>02 · Analyze text</h3>
            <p>TF-IDF converts text into numerical features.</p>
            <h3>03 · View prediction</h3>
            <p>The trained classifier predicts a label.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
            <p class="small-label">IMPORTANT NOTICE</p>
            <p>This tool recognizes patterns learned from
            labeled articles. It does not independently
            verify claims or their sources.</p>
        </div>
        """, unsafe_allow_html=True)

    # ---------- RESULT CARD ----------
    result = st.session_state.result

    if result:
        st.divider()
        st.markdown("## 📊 Analysis result")

        r1, r2 = st.columns(2)

        with r1:
            if result["prediction"] == "Fake":
                st.error("Prediction: Likely Fake News")
            else:
                st.success("Prediction: Likely Real News")

        with r2:
            st.metric(
                "Confidence in predicted class",
                f'{result["confidence"]:.1f}%'
            )

        st.caption(
            "Model confidence is not the probability that "
            "the news claim is true."
        )

        st.markdown("### Class scores")

        for label, score in result["scores"].items():
            st.write(f"*{label}* — {score:.1%}")
            st.progress(score)

# ---------- HISTORY ----------
elif page == "Prediction History":
    st.markdown("## 📚 Prediction history")
    st.write("Review news analyzed during this browser session.")

    if st.session_state.history:
        st.dataframe(
            st.session_state.history,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear prediction history"):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()
    else:
        st.info("No predictions yet. Analyze news to create history.")

# ---------- ABOUT ----------
elif page == "About Project":
    st.markdown("""
    <div class="hero">
        <h1>About TruthLens AI</h1>
        <p>AD3501 Mini Project · Fake News Detection
        Using Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Project objective")
    st.write(
        "Develop a web application that classifies news text "
        "using a supervised machine learning model."
    )

    st.markdown("### Technologies")
    st.write("- Python")
    st.write("- Pandas")
    st.write("- Scikit-learn")
    st.write("- TF-IDF text vectorization")
    st.write("- Logistic Regression")
    st.write("- Streamlit")

    st.markdown("### Limitations")
    st.write(
        "The prediction depends on the training data and "
        "may be incorrect. Independent source verification "
        "is necessary before deciding whether a claim is true."
    )

# ---------- FOOTER ----------
st.divider()
st.caption(
    "TruthLens AI · AD3501 Mini Project · "
    "Machine learning supports analysis; it does not replace fact-checking."
)