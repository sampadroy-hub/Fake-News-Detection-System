import pandas as pd
import pickle
from preprocess import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# =========================
# 1. LOAD DATA
# =========================
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

true_df['label'] = 0
fake_df['label'] = 1

df = pd.concat([true_df, fake_df], axis=0)

# Shuffle full dataset (NO LIMIT)
df = df.sample(frac=1).reset_index(drop=True)

# Combine text
df['content'] = df['title'] + " " + df['text']

# =========================
# 2. PREPROCESSING (IMPORTANT)
# =========================
df['content'] = df['content'].apply(clean_text)

# =========================
# 3. FEATURE EXTRACTION
# =========================
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['content'])
y = df['label']

# =========================
# 4. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. MODELS
# =========================
lr = LogisticRegression(max_iter=500)
nb = MultinomialNB()
svm = SVC(probability=True, kernel='linear')
rf = RandomForestClassifier(n_estimators=100)

# =========================
# 6. TRAIN INDIVIDUAL MODELS
# =========================
models = {
    "Logistic Regression": lr,
    "Naive Bayes": nb,
    "SVM": svm,
    "Random Forest": rf
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }

    print(f"\n{name} Report:")
    print(classification_report(y_test, y_pred))

# =========================
# 7. VOTING (HYBRID ML MODEL)
# =========================
voting_model = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('nb', nb),
        ('rf', rf)
    ],
    voting='soft'
)

voting_model.fit(X_train, y_train)
y_pred_voting = voting_model.predict(X_test)

results["Voting Classifier"] = {
    "Accuracy": accuracy_score(y_test, y_pred_voting),
    "Precision": precision_score(y_test, y_pred_voting),
    "Recall": recall_score(y_test, y_pred_voting),
    "F1": f1_score(y_test, y_pred_voting)
}

print("\nVoting Classifier Report:")
print(classification_report(y_test, y_pred_voting))

# =========================
# 8. PRINT COMPARISON TABLE
# =========================
print("\n=== MODEL COMPARISON ===")
for model_name, metrics in results.items():
    print(f"\n{model_name}")
    for metric, value in metrics.items():
        print(f"{metric}: {round(value, 4)}")

# =========================
# 9. SAVE BEST MODEL
# =========================
pickle.dump(voting_model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))