from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data (small example for demo purposes)
texts = [
    'Win a $1000 prize now!',
    'Limited offer, click here!',
    'Hi, how are you?',
    'Let’s catch up tomorrow.',
    'Free entry in a contest!',
    'Congratulations, you won!',
    'Are we still meeting today?',
    'Call me when you are free.'
]
labels = [1, 1, 0, 0, 1, 1, 0, 0]  # 1 = Spam, 0 = Not Spam

# Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X, labels)

# Take user input
user_input = input("Enter a message to check for spam: ")
user_X = vectorizer.transform([user_input])

# Predict
prediction = model.predict(user_X)[0]

# Output result
print("\nPrediction:")
if prediction == 1:
    print("This message is classified as SPAM 🚫")
else:
    print("This message is NOT SPAM ✅")
