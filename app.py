import streamlit as st
import pickle
import numpy as np

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hero section */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0;
}

/* Card */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
}

/* Result badges */
.result-positive {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    color: #6ee7b7;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
}
.result-negative {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #f87171;
    color: #fca5a5;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
}
.result-neutral {
    background: linear-gradient(135deg, #1e3a5f, #1e40af);
    border: 1px solid #60a5fa;
    color: #93c5fd;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
}
.result-irrelevant {
    background: linear-gradient(135deg, #1f2937, #374151);
    border: 1px solid #9ca3af;
    color: #d1d5db;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
}

.confidence-label {
    color: #94a3b8;
    font-size: 0.85rem;
    text-align: center;
    margin-top: 0.5rem;
}
.confidence-value {
    color: #e2e8f0;
    font-size: 1rem;
    font-weight: 600;
    text-align: center;
}

/* Textarea styling */
textarea,
.stTextArea textarea,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="base-input"] textarea {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    caret-color: #a78bfa !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
}
textarea::placeholder,
.stTextArea textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: opacity 0.2s ease !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}

/* History table */
.history-row {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid rgba(255,255,255,0.07);
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.08) !important;
}

/* Label color override */
label, .stTextArea label {
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, vectorizer, le

model, vectorizer, le = load_models()

# ─── Sentiment Helper ──────────────────────────────────────────────────────────
SENTIMENT_CONFIG = {
    "Positive":    {"emoji": "😊", "css": "result-positive"},
    "Negative":    {"emoji": "😞", "css": "result-negative"},
    "Neutral":     {"emoji": "😐", "css": "result-neutral"},
    "Irrelevant":  {"emoji": "🤷", "css": "result-irrelevant"},
}

def predict_sentiment(text: str):
    vec = vectorizer.transform([text])
    pred_idx = model.predict(vec)[0]
    label = le.inverse_transform([pred_idx])[0]

    # Confidence via decision_function or predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = float(np.max(proba)) * 100
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(vec)[0]
        exp_scores = np.exp(scores - np.max(scores))
        proba = exp_scores / exp_scores.sum()
        confidence = float(np.max(proba)) * 100
    else:
        confidence = None

    return label, confidence

# ─── Session State ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🐦 Twitter Sentiment Analyzer</h1>
    <p>Powered by Machine Learning · Analyze tweet sentiment instantly</p>
</div>
""", unsafe_allow_html=True)

# ─── Input Card ────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
tweet_text = st.text_area(
    "Enter a tweet or any text to analyze:",
    placeholder="e.g. I absolutely love using this product! It changed my life. 🚀",
    height=130,
    key="tweet_input",
)
analyze_btn = st.button("✨ Analyze Sentiment")
st.markdown('</div>', unsafe_allow_html=True)

# ─── Prediction ────────────────────────────────────────────────────────────────
if analyze_btn:
    if not tweet_text.strip():
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing sentiment..."):
            label, confidence = predict_sentiment(tweet_text.strip())

        cfg = SENTIMENT_CONFIG.get(label, {"emoji": "🔍", "css": "result-neutral"})
        emoji = cfg["emoji"]
        css_class = cfg["css"]

        st.markdown(f"""
<div class="{css_class}">
    {emoji} &nbsp; {label}
</div>
""", unsafe_allow_html=True)

        if confidence is not None:
            st.markdown(f"""
<p class="confidence-label">Model Confidence</p>
<p class="confidence-value">{confidence:.1f}%</p>
""", unsafe_allow_html=True)
            st.progress(int(confidence))

        # Save to history
        st.session_state.history.insert(0, {
            "text": tweet_text.strip()[:60] + ("…" if len(tweet_text.strip()) > 60 else ""),
            "label": label,
            "emoji": emoji,
            "confidence": confidence,
        })

# ─── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("#### 🕓 Recent Analyses")
    for item in st.session_state.history[:8]:
        conf_str = f"{item['confidence']:.1f}%" if item["confidence"] is not None else "—"
        st.markdown(f"""
<div class="history-row">
    <span style="color:#e2e8f0;font-size:0.9rem;">"{item['text']}"</span>
    <span style="color:#94a3b8;font-size:0.85rem;">{item['emoji']} {item['label']} &nbsp;|&nbsp; {conf_str}</span>
</div>
""", unsafe_allow_html=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<br>
<p style="text-align:center;color:#475569;font-size:0.8rem;">
    Twitter Sentiment Analyzer · Built with Streamlit & scikit-learn
</p>
""", unsafe_allow_html=True)