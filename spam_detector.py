import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv(r"C:\Users\prade\Downloads\mail_data.csv", encoding="latin-1")
print(df.columns)
print(df.head())

# Keep only required columns
df = df[["Category", "Message"]]

# Rename columns
df.columns = ["label", "email"]

# Display dataset information
print("===== DATASET INFORMATION =====")
print("Total Messages:", len(df))
print("\nFirst 5 Records:")
print(df.head())

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
print("\n===== MODEL EVALUATION =====")

print("\nActual Labels:")
print(list(y_test[:10]))

print("\nPredicted Labels:")
print(list(predictions[:10]))

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

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
