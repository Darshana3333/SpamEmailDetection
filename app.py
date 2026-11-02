
import csv
import io
import json
import pickle
from pathlib import Path
from datetime import datetime

import streamlit as st

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "spam_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="SpamGuard AI | Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# DARK THEME CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263247;
}
[data-testid="stSidebar"] * {
    color: #e5e7eb;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
[data-testid="stMetric"] {
    background: #151f32;
    border: 1px solid #29364c;
    padding: 18px;
    border-radius: 14px;
}
[data-testid="stMetricLabel"] {
    color: #a5b4fc;
}
[data-testid="stMetricValue"] {
    color: #ffffff;
}
div.stButton > button {
    background: #6366f1;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0.55rem 1rem;
    font-weight: 600;
}
div.stButton > button:hover {
    background: #4f46e5;
    color: white;
    border: none;
}
div[data-testid="stTextArea"] textarea {
    background: #151f32;
    color: white;
    border: 1px solid #374151;
    border-radius: 10px;
}
div[data-testid="stAlert"] {
    border-radius: 10px;
}
h1, h2, h3 {
    color: #f8fafc;
}
.small-muted {
    color: #94a3b8;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL AND METRICS
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    metrics = load_metrics()
except FileNotFoundError as error:
    st.error(f"Required file not found: {error.filename}")
    st.info(
        "Keep app.py, spam_model.pkl and metrics.json "
        "inside the same project folder."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load the model or metrics: {error}")
    st.stop()

# --------------------------------------------------
# SESSION HISTORY
# --------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

history = st.session_state.prediction_history

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("# 📧 SpamGuard AI")
    st.caption("Machine Learning Dashboard")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Spam Detector",
            "Prediction History",
            "About Project",
        ],
    )

    st.divider()
    st.markdown("**MODEL INFORMATION**")
    st.caption("TF-IDF + Multinomial Naive Bayes")
    st.caption("Dataset: SMS Spam Collection")
    st.caption("Mode: Local application")

# --------------------------------------------------
# HELPER: METRIC CARD
# --------------------------------------------------
def show_header(title, subtitle):
    st.title(title)
    st.markdown(
        f'<p class="small-muted">{subtitle}</p>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# OVERVIEW DASHBOARD
# --------------------------------------------------
if page == "Overview":
    show_header(
        "📊 Dashboard Overview",
        "Monitor model performance and explore spam detection results.",
    )

    accuracy = float(metrics.get("accuracy", 0))
    precision = float(metrics.get("spam_precision", 0))
    recall = float(metrics.get("spam_recall", 0))
    f1_score = float(metrics.get("spam_f1", 0))
    test_messages = int(metrics.get("test_messages", 0))

    st.subheader("Model Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy", f"{accuracy:.2%}")
    c2.metric("Spam Precision", f"{precision:.2%}")
    c3.metric("Spam Recall", f"{recall:.2%}")
    c4.metric("Spam F1-Score", f"{f1_score:.2%}")

    st.caption(
        f"Evaluation on {test_messages:,} held-out test messages."
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("📈 Evaluation Metrics")
        metric_chart = {
            "Score": [
                accuracy * 100,
                precision * 100,
                recall * 100,
                f1_score * 100,
            ]
        }
        st.bar_chart(
            metric_chart,
            horizontal=True,
            x_label="Metric",
            y_label="Percentage (%)",
        )
        st.caption(
            "Accuracy, precision, recall and F1-score are shown as percentages."
        )

    with right:
        st.subheader("🧪 Confusion Matrix")
        matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])

        st.dataframe(
            {
                "Actual Class": ["Not Spam", "Spam"],
                "Predicted Not Spam": [matrix[0][0], matrix[1][0]],
                "Predicted Spam": [matrix[0][1], matrix[1][1]],
            },
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Rows represent actual labels; columns represent predictions."
        )

    st.divider()
    st.subheader("📬 Your Testing Activity")

    spam_count = sum(
        1 for item in history if item["Prediction"] == "SPAM"
    )
    ham_count = sum(
        1 for item in history if item["Prediction"] == "NOT SPAM"
    )

    a, b, c = st.columns(3)
    a.metric("Messages Tested This Session", len(history))
    b.metric("Predicted Spam", spam_count)
    c.metric("Predicted Not Spam", ham_count)

    if history:
        st.markdown("**Prediction distribution — current session**")
        st.bar_chart(
            {
                "Prediction": ["Not Spam", "Spam"],
                "Count": [ham_count, spam_count],
            },
            x="Prediction",
            y="Count",
        )
    else:
        st.info(
            "No messages tested in this session yet. "
            "Open Spam Detector to start testing."
        )

# --------------------------------------------------
# SPAM DETECTOR
# --------------------------------------------------
elif page == "Spam Detector":
    show_header(
        "🔍 Spam Message Detector",
        "Enter an SMS-style message and let the trained model classify it.",
    )

    st.info(
        "This model was trained on an SMS dataset. "
        "It may not perform equally well on every type of email."
    )

    with st.form("spam_detection_form"):
        message = st.text_area(
            "Message to analyze",
            placeholder=(
                "Example: Congratulations! You have won a free prize. Claim now!"
            ),
            height=160,
        )
        submitted = st.form_submit_button(
            "🔎 Detect Spam",
            use_container_width=True,
        )

    if submitted:
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            prediction = str(model.predict([message])[0]).lower()
            probabilities = model.predict_proba([message])[0]
            classes = [str(item).lower() for item in model.classes_]

            if "spam" in classes:
                spam_probability = float(
                    probabilities[classes.index("spam")]
                )
            else:
                spam_probability = 0.0

            is_spam = prediction == "spam"
            label = "SPAM" if is_spam else "NOT SPAM"

            if is_spam:
                st.error("🚨 Prediction: SPAM")
            else:
                st.success("✅ Prediction: NOT SPAM")

            p1, p2 = st.columns(2)
            p1.metric("Estimated Spam Probability", f"{spam_probability:.1%}")
            p2.metric("Prediction", label)

            st.progress(
                max(0.0, min(1.0, spam_probability)),
                text="Estimated spam probability",
            )

            st.caption(
                "This probability is the model's estimate, "
                "not a guarantee that the message is spam."
            )

            # Add each submitted prediction to this session's history.
            st.session_state.prediction_history.insert(
                0,
                {
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Message": message.strip(),
                    "Prediction": label,
                    "Spam Probability": round(spam_probability * 100, 2),
                },
            )

            st.toast("Prediction saved to this session's history.")

# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------
elif page == "Prediction History":
    show_header(
        "🕘 Prediction History",
        "Review and download messages tested during this session.",
    )

    if history:
        st.metric("Total Messages Tested", len(history))

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

        csv_buffer = io.StringIO()
        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=[
                "Time",
                "Message",
                "Prediction",
                "Spam Probability",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

        st.download_button(
            label="⬇️ Download History as CSV",
            data=csv_buffer.getvalue().encode("utf-8-sig"),
            file_name="spam_detection_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info(
            "History is empty. Go to Spam Detector and analyze a message first."
        )

# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------
elif page == "About Project":
    show_header(
        "ℹ️ About This Project",
        "A machine learning project for classifying SMS-style messages.",
    )

    st.subheader("Project Objective")
    st.write(
        "The application predicts whether a message resembles spam "
        "or not spam using a trained text-classification model."
    )

    st.subheader("Technologies Used")
    st.markdown("""
    - **Python** — programming language
    - **Streamlit** — interactive web dashboard
    - **Scikit-learn** — TF-IDF and Naive Bayes model
    - **CSV** — dataset and downloadable prediction history
    - **JSON** — saved evaluation metrics
    """)

    st.subheader("How It Works")
    st.markdown("""
    1. The training script loads labeled SMS messages.
    2. TF-IDF converts text into numerical features.
    3. Multinomial Naive Bayes learns patterns in the messages.
    4. The trained pipeline is saved as `spam_model.pkl`.
    5. This dashboard loads the model and predicts new messages.
    """)

    st.warning(
        "Educational project: predictions can be incorrect. "
        "Do not rely on this model as the only spam protection."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "SpamGuard AI • SMS Spam Detection • "
    "TF-IDF + Multinomial Naive Bayes"
)
