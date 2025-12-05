import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from joblib import dump

# Paths
BASE = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE, "data", "merged_sql_10000.csv")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    df["query"] = df["query"].astype(str).str.strip()
    df = df[df["query"] != ""]

    X = df["query"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training SVM classifier...")
    clf = LinearSVC()
    clf.fit(X_train_tfidf, y_train)

    print("\n=== CLASSIFICATION REPORT ===")
    y_pred = clf.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))

    print("Saving models...")
    dump(vectorizer, os.path.join(MODEL_DIR, "tfidf.joblib"))
    dump(clf, os.path.join(MODEL_DIR, "svm.joblib"))

    print("\nDONE. Models saved in /models")

if __name__ == "__main__":
    main()
