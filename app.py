import streamlit as st
import joblib
import numpy as np
import re
import random
from scipy.sparse import hstack

# ===============================
# Load model & vectorizer
# ===============================
model = joblib.load("models/detector.pkl")
tfidf = joblib.load("models/tfidf.pkl")

# ===============================
# Stylometric feature extraction
# ===============================
def extract_features(text):
    words = re.findall(r"\w+", text.lower())
    sentences = re.split(r"[.!?]", text)

    return np.array([
        len(words),                                  # word count
        len(sentences),                              # sentence count
        np.mean([len(w) for w in words]) if words else 0,  # avg word length
        len(set(words)) / len(words) if words else 0,      # lexical diversity
        len(re.findall(r"[,.!?;:]", text)) / max(len(text), 1),
        sum(1 for c in text if c.isupper()) / max(len(text), 1)
    ])

# ===============================
# Humanize Text Module
# ===============================
def humanize_text(text):
    import random
    import re

    fillers = [
        "honestly",
        "to be fair",
        "in a way",
        "from my point of view",
        "I think",
        "it feels like"
    ]

    casual_words = {
        "therefore": "so",
        "however": "but",
        "moreover": "also",
        "utilize": "use",
        "significant": "important",
        "numerous": "many",
        "individuals": "people"
    }

    sentences = re.split(r'(?<=[.!?]) +', text)
    new_sentences = []

    for s in sentences:
        words = s.split()

        # 1️⃣ Casser les phrases longues
        if len(words) > 25 and random.random() < 0.6:
            mid = len(words) // 2
            s = " ".join(words[:mid]) + ". " + " ".join(words[mid:])

        # 2️⃣ Remplacement lexical moins académique
        for k, v in casual_words.items():
            if random.random() < 0.5:
                s = re.sub(rf"\b{k}\b", v, s, flags=re.IGNORECASE)

        # 3️⃣ Ajouter hésitation humaine
        if random.random() < 0.25:
            s = random.choice(fillers).capitalize() + ", " + s.lower()

        # 4️⃣ Supprimer ponctuation excessive
        s = re.sub(r";", ",", s)

        new_sentences.append(s.strip())

    return " ".join(new_sentences)


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="AI vs Human Text Detector", layout="centered")

st.title("🤖 AI vs Human Text Detector")
st.markdown("""
### 🔍 Project Description
This application detects whether a given text is **AI-generated** or **human-written**
using a combination of:
- **TF-IDF (lexical features)**
- **Stylometric features (writing style)**

It also includes a **Humanize Text** module that rewrites AI-generated text
to make it **more natural and human-like**.
""")

st.divider()

# ===============================
# Text Input
# ===============================
text = st.text_area(
    "✍️ Enter your text below:",
    height=200,
    placeholder="Paste an essay, paragraph, or AI-generated text here..."
)

# ===============================
# Analyze Button
# ===============================
if st.button("🔍 Analyze Text"):
    if len(text.strip()) < 50:
        st.warning("Please enter a longer text (at least 50 characters).")
    else:
        X_tfidf = tfidf.transform([text])
        X_style = extract_features(text).reshape(1, -1)
        X = hstack([X_tfidf, X_style])

        probas = model.predict_proba(X)[0]
        pred = probas.argmax()
        confidence = probas[pred]

        if pred == 1:
            st.error(f"🤖 **AI Generated Text**  Confidence: **{confidence:.2%}**")
        else:
            st.success(f"✍️ **Human Written Text**  Confidence: **{confidence:.2%}**")

        st.caption("Prediction based on writing style + lexical patterns.")

st.divider()

# ===============================
# Humanization Section
# ===============================
st.subheader("🧠 Humanize Text")
st.markdown("""
This module rewrites text to reduce **AI-like patterns**, such as:
- Overly formal vocabulary
- Long, structured sentences
- Repetitive phrasing

⚠️ This is **not plagiarism**, but **style adaptation**.
""")

if st.button("✨ Humanize Text"):
    if len(text.strip()) < 50:
        st.warning("Please enter a longer text first.")
    else:
        humanized = humanize_text(text)
        st.text_area("📝 Humanized Version:", humanized, height=200)

st.caption("Made with classical Machine Learning — no deep learning models.")
