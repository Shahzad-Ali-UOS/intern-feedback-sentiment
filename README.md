# 💬 Intern Feedback Sentiment & Satisfaction Analytics Engine

An end-to-end NLP-powered sentiment classification and feedback triage engine built for internship platforms. The system automatically categorizes qualitative intern reviews into **Positive**, **Neutral**, and **Negative** sentiment classes, visualizes real-time confidence scores via an interactive speedometer gauge meter, and identifies operational friction points across core program dimensions.

---

## 📌 Project Overview

Collecting qualitative intern reviews often leads to unstructured feedback backlogs. This platform automates:
- **Real-Time Sentiment Classification:** Predicts whether an incoming review is Positive, Neutral, or Negative using sublinear TF-IDF word/n-gram features and regularized Logistic Regression.
- **Speedometer Sentiment Meter:** Dynamically positions a gauge needle and updates status badges based on probabilistic confidence.
- **Operational Area Triage:** Groups negative feedback across key internship touchpoints (**Mentorship**, **Task Clarity**, **Portal/LMS**, **Curriculum**, and **Workload/Pacing**) to highlight areas requiring immediate administrative intervention.
- **Executive Improvement Matrix:** Provides actionable remediation strategies targeted at common intern dissatisfaction drivers.

---

## 🛠️ Tech Stack & Architecture

- **Language:** Python 3.10+
- **Machine Learning & NLP:** Scikit-Learn (`TfidfVectorizer`, `LogisticRegression`)
- **Data Engineering:** Pandas, NumPy
- **Dashboard & Visualizations:** Streamlit, Plotly (`graph_objects`, `express`)
- **Artifact Serialization:** Joblib

---

## 📂 Repository Structure

```text
intern-feedback-sentiment/
├── data/
│   └── feedback_data.csv          # Synthesized intern feedback dataset
├── models/
│   ├── tfidf_vectorizer.pkl       # Fitted N-gram TF-IDF vectorizer artifact
│   └── sentiment_model.pkl        # Trained multiclass Logistic Regression model
├── app.py                         # Interactive Streamlit dashboard with gauge meter
├── data_generator.py              # Realistic feedback dataset generator
├── train.py                       # Model training, validation, and artifact export
├── requirements.txt               # Application dependencies
└── README.md                      # Project documentation
