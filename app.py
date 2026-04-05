import time
import re
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# ------------------ COUNTERS ------------------
total_count = 0
fake_count = 0
real_count = 0
# ---------------------------------------------

# Load dataset
df = pd.read_csv("data/fake_job_postings.csv")

# Fill missing values
df['description'] = df['description'].fillna('')
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)   # remove special characters
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text
df['description'] = df['description'].apply(clean_text)
from sklearn.feature_extraction.text import TfidfVectorizer

# Input & Output
X = df['description']
y = df['fraudulent']

# Convert text to numbers
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=7000,
    ngram_range=(1,2)
)
X_vectorized = vectorizer.fit_transform(X)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report

# Predictions on test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Detailed report
print(classification_report(y_test, y_pred))


# Home route
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global total_count, fake_count, real_count

    job_description = request.form.get("job_description")

    # Empty input handling
    if job_description is None or job_description.strip() == "":
        return render_template(
            "result.html",
            prediction="⚠ Please enter a job description",
            confidence=0,
            reasons=[],
            total=total_count,
            fake=fake_count,
            real=real_count
        )

    # Clean input
    job_description = clean_text(job_description)

    time.sleep(1)

    # Convert input text → vector
    input_data = vectorizer.transform([job_description])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Confidence
    probability = model.predict_proba(input_data)[0]
    confidence = round(max(probability) * 100, 2)

    # Reason detection
    reasons = []
    suspicious_words = [
        "earn fast", "no experience", "work from home", "investment",
        "urgent hiring", "urgent requirement", "no interview",
        "no interviews", "quick money", "limited seats",
        "apply fast", "registration fee", "earn daily",
        "click here", "whatsapp job"
    ]

    for word in suspicious_words:
        if word in job_description:
            reasons.append(word)

    # Counter update
    total_count += 1

    # ✅ SMART DECISION LOGIC
    if len(reasons) >= 2:
        result = "⚠ Fake Job Detected"
        fake_count += 1

    elif prediction == 1 and confidence > 70:
        result = "⚠ Fake Job Detected"
        fake_count += 1

    else:
        result = "✅ Job Looks Real"
        real_count += 1

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence,
        reasons=reasons,
        total=total_count,
        fake=fake_count,
        real=real_count
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True)