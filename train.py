import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

def train_sentiment_model():
    data_path = os.path.join("data", "feedback_data.csv")
    if not os.path.exists(data_path):
        from data_generator import generate_feedback_dataset
        generate_feedback_dataset()

    df = pd.read_csv(data_path)
    X = df['review_text']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 1. Feature Extraction: N-gram TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words='english',
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 2. Classifier: Multiclass Logistic Regression
    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=1.2,
        random_state=42
    )
    clf.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print("\n" + "="*50)
    print("      SENTIMENT CLASSIFICATION BENCHMARK")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}\n")
    print(classification_report(y_test, y_pred))

    # 3. Export Artifacts
    os.makedirs("models", exist_ok=True)
    vec_path = os.path.join("models", "tfidf_vectorizer.pkl")
    model_path = os.path.join("models", "sentiment_model.pkl")

    joblib.dump(vectorizer, vec_path)
    joblib.dump({
        'model': clf,
        'classes': list(clf.classes_),
        'metrics': {'accuracy': acc, 'macro_f1': macro_f1}
    }, model_path)

    print(f"[OK] Vectorizer exported -> {vec_path}")
    print(f"[OK] Model artifact exported -> {model_path}")

if __name__ == "__main__":
    train_sentiment_model()