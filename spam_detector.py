import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample Dataset
data = {
    "email": [
        "Win a free iPhone now",
        "Meeting at 3 PM today",
        "Congratulations! You won a lottery",
        "Project submission tomorrow",
        "Claim your free gift card",
        "Let's discuss the assignment",
        "Earn money quickly from home",
        "Your account has been credited",
        "Free vacation package available",
        "Team meeting scheduled for Monday"
    ],
    "label": [
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    df["email"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# Convert text into numerical features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
predictions = model.predict(X_test_vec)

# Display Results
print("===== MODEL EVALUATION =====")
print("Actual Labels   :", list(y_test))
print("Predicted Labels:", list(predictions))

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# User Input Prediction
print("\n===== SPAM EMAIL CHECKER =====")
user_email = input("Enter an email message: ")

user_email_vec = vectorizer.transform([user_email])
result = model.predict(user_email_vec)[0]

print("\nPrediction:", result.upper())

if result == "spam":
    print("Warning: This email is likely SPAM.")
else:
    print("This email appears to be HAM (Not Spam).")