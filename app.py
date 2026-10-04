import os
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Intern Feedback Sentiment & Improvement Engine",
    page_icon="💬",
    layout="wide"
)

# Custom Styling for Smooth Animations and Emojis
st.markdown("""
<style>
    @keyframes pulseCard {
        0% { transform: scale(0.98); opacity: 0.8; }
        50% { transform: scale(1.01); opacity: 1; }
        100% { transform: scale(1.0); opacity: 1; }
    }
    .sentiment-badge {
        animation: pulseCard 0.6s ease-out;
        padding: 18px 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-weight: 700;
        font-size: 1.35rem;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .bg-positive { background: linear-gradient(135deg, #10B981, #059669); }
    .bg-neutral { background: linear-gradient(135deg, #F59E0B, #D97706); }
    .bg-negative { background: linear-gradient(135deg, #EF4444, #DC2626); }
    .emoji-hero {
        font-size: 3.5rem;
        display: block;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("💬 Intern Feedback Sentiment & Satisfaction Analytics")
st.caption("AI-powered NLP triage classifying candidate sentiment across 3 segmented operational zones.")

# Load Artifacts
vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
model_path = os.path.join("models", "sentiment_model.pkl")

@st.cache_resource
def load_artifacts():
    if not (os.path.exists(vectorizer_path) and os.path.exists(model_path)):
        return None, None
    vec = joblib.load(vectorizer_path)
    model_data = joblib.load(model_path)
    return vec, model_data

vectorizer, model_data = load_artifacts()

if vectorizer is None or model_data is None:
    st.error("⚠️ Model artifacts missing. Run `python train.py` first.")
    st.stop()

clf = model_data['model']
classes = model_data['classes']
metrics = model_data['metrics']

# Sidebar
st.sidebar.header("📊 Model Metrics")
st.sidebar.success("Model: TF-IDF + Logistic Regression")
st.sidebar.markdown(f"- **Accuracy:** `{metrics['accuracy']:.2%}`")
st.sidebar.markdown(f"- **Macro F1:** `{metrics['macro_f1']:.4f}`")
st.sidebar.markdown("---")
st.sidebar.markdown("**Target Sentiment Zones:**")
st.sidebar.markdown("🔴 **Negative:** `0.0 – 33.3`")
st.sidebar.markdown("🟡 **Neutral:** `33.3 – 66.6`")
st.sidebar.markdown("🟢 **Positive:** `66.6 – 100.0`")

# Helper Function: Render Speedometer Gauge
def create_sentiment_speedometer(score_val, sentiment_label):
    """
    Renders a 3-category semi-circle gauge meter with pointer needle.
    Negative: 0 - 33.3 | Neutral: 33.3 - 66.7 | Positive: 66.7 - 100
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_val,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 32, 'family': 'Arial'}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': "#475569",
                'tickvals': [16.6, 50, 83.3],
                'ticktext': ['NEGATIVE', 'NEUTRAL', 'POSITIVE'],
                'tickfont': {'size': 13, 'weight': 'bold'}
            },
            'bar': {'color': '#1E293B', 'thickness': 0.28},  # Pointer / Needle color
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#CBD5E1",
            'steps': [
                {'range': [0, 33.33], 'color': '#EF4444'},     # Negative Red
                {'range': [33.33, 66.66], 'color': '#FBBF24'}, # Neutral Yellow/Amber
                {'range': [66.66, 100.0], 'color': '#10B981'}  # Positive Green
            ],
            'threshold': {
                'line': {'color': "#0F172A", 'width': 5},
                'thickness': 0.85,
                'value': score_val
            }
        }
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#1E293B", 'family': "sans-serif"}
    )
    return fig

# Main Navigation
tab_live, tab_cohort = st.tabs(["🔍 Real-Time Review Gauge", "📈 Cohort Satisfaction & Action Areas"])

with tab_live:
    st.subheader("Analyze Intern Review")
    user_input = st.text_area(
        "Enter intern feedback below to test real-time classification:",
        value="The mentors were very slow to respond to portal questions, and the assignment instructions were quite confusing.",
        height=100
    )

    if st.button("Evaluate Sentiment", type="primary"):
        if user_input.strip():
            vec_input = vectorizer.transform([user_input])
            pred = clf.predict(vec_input)[0]
            probs = clf.predict_proba(vec_input)[0]
            prob_dict = dict(zip(classes, probs))

            # Continuous Sentiment Index:
            # Map Negative (0), Neutral (50), Positive (100) weighted by probabilities
            pos_p = prob_dict.get('Positive', 0.0)
            neu_p = prob_dict.get('Neutral', 0.0)
            neg_p = prob_dict.get('Negative', 0.0)
            calculated_gauge_score = round((neg_p * 0.0) + (neu_p * 50.0) + (pos_p * 100.0), 1)

            col_badge, col_gauge = st.columns([1.1, 1.9])

            with col_badge:
                st.write("")
                if pred == 'Positive':
                    st.markdown("""
                        <div class="sentiment-badge bg-positive">
                            <span class="emoji-hero">😊</span>
                            POSITIVE SENTIMENT
                            <div style="font-size: 0.95rem; font-weight: normal; margin-top: 8px;">High Satisfaction & Engagement</div>
                        </div>
                    """, unsafe_allow_html=True)
                elif pred == 'Neutral':
                    st.markdown("""
                        <div class="sentiment-badge bg-neutral">
                            <span class="emoji-hero">😐</span>
                            NEUTRAL SENTIMENT
                            <div style="font-size: 0.95rem; font-weight: normal; margin-top: 8px;">Acceptable / Baseline Experience</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="sentiment-badge bg-negative">
                            <span class="emoji-hero">😞</span>
                            NEGATIVE SENTIMENT
                            <div style="font-size: 0.95rem; font-weight: normal; margin-top: 8px;">Actionable Intervention Required</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("##### Prediction Probabilities")
                for label, color in [('Positive', '#10B981'), ('Neutral', '#F59E0B'), ('Negative', '#EF4444')]:
                    p_val = prob_dict.get(label, 0.0)
                    st.write(f"**{label}:** `{p_val:.1%}`")
                    st.progress(float(p_val))

            with col_gauge:
                st.markdown("<h4 style='text-align: center;'>Sentiment Index Meter</h4>", unsafe_allow_html=True)
                fig_meter = create_sentiment_speedometer(calculated_gauge_score, pred)
                st.plotly_chart(fig_meter, use_container_width=True)

with tab_cohort:
    st.subheader("Cohort-Wide Feedback Breakdown")
    data_path = os.path.join("data", "feedback_data.csv")
    if os.path.exists(data_path):
        df_cohort = pd.read_csv(data_path)

        c1, c2 = st.columns(2)

        with c1:
            sentiment_counts = df_cohort['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            fig_pie = px.pie(
                sentiment_counts, names='Sentiment', values='Count',
                title="Overall Sentiment Distribution",
                color='Sentiment',
                color_discrete_map={'Positive': '#10B981', 'Neutral': '#F59E0B', 'Negative': '#EF4444'},
                hole=0.45
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            neg_df = df_cohort[df_cohort['sentiment'] == 'Negative']
            cat_counts = neg_df['category'].value_counts().reset_index()
            cat_counts.columns = ['Operational Area', 'Negative Reviews']

            fig_bar = px.bar(
                cat_counts, x='Negative Reviews', y='Operational Area',
                orientation='h',
                title="Top Sources of Intern Dissatisfaction",
                color='Negative Reviews',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.subheader("🎯 Executive Action Areas for Program Improvement")

        col_a, col_b = st.columns(2)
        with col_a:
            st.error("🚨 Critical Focus: Task Clarity & Mentorship Latency")
            st.write(
                "Feedback indicates interns struggle most with vague requirement documentation "
                "and delayed mentor responses during bug resolution. Implement standardized PR templates "
                "and enforce a 24-hour SLA on mentor queries."
            )
        with col_b:
            st.warning("⚠️ Secondary Focus: Portal Stability & Workload Pacing")
            st.write(
                "Submission upload timeouts near deadlines cause high friction. "
                "Scale backend server workers during submission cutoffs and avoid back-to-back sprint deadlines."
            )
    else:
        st.info("Run `python data_generator.py` to generate the cohort dataset.")