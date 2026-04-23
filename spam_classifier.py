# Spam Classifier Project

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels into numbers (ham=0, spam=1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42
)

# Convert text into numerical vectors
vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_transformed, y_train)

# Make predictions
y_pred = model.predict(X_test_transformed)

# Print accuracy
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

# Detailed report (extra marks)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Show some sample predictions
print("\nSample Predictions:\n")
for i in range(5):
    print("Message:", X_test.iloc[i])
    print("Predicted:", "Spam" if y_pred[i] == 1 else "Not Spam")
    print("Actual:", "Spam" if y_test.iloc[i] == 1 else "Not Spam")
    print("-" * 50)

# ----------- User Input Section -----------
print("\n📩 Enter your own messages to test (type 'exit' to stop):")

while True:
    user_input = input("\nEnter message: ")

    if user_input.lower() == 'exit':
        print("Exiting...")
        break

    # Transform input text
    user_input_transformed = vectorizer.transform([user_input])

    # Predict
    prediction = model.predict(user_input_transformed)

    if prediction[0] == 1:
        print("🚨 Spam Message")
    else:
        print("✅ Not Spam")