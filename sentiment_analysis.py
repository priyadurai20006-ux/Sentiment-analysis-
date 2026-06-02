# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Create Dataset
data = {
    'Review': [
        'This product is amazing',
        'Worst product ever',
        'I love this phone',
        'Very bad quality',
        'Excellent item',
        'Not good',
        'Best purchase',
        'Waste of money',
        'Super quality',
        'Very disappointed'
    ],
    'Sentiment': [
        'Positive', 'Negative', 'Positive', 'Negative',
        'Positive', 'Negative', 'Positive', 'Negative',
        'Positive', 'Negative'
    ]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df['Review'], df['Sentiment'], test_size=0.2, random_state=42
)

# Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction
prediction = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy:", accuracy)

# Test with new sentence
test = ["This phone is very good"]
test_vec = vectorizer.transform(test)

print("\nTest Result:", model.predict(test_vec))

Dataset:

               Review Sentiment
0   This product is amazing   Positive
1      Worst product ever   Negative
2       I love this phone   Positive
3        Very bad quality   Negative
4          Excellent item  Positive
5                Not good  Negative
6          Best purchase   Positive
7         Waste of money   Negative
8           Super quality Positive
9       Very disappointed Negative

Accuracy: 0.0

Test Result: ['Positive']