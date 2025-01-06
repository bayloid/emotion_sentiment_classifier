#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # silence TensorFlow messages
print(sys.version)


# In[2]:


# Imports

print("Importing libraries")

# NLTK imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify import ConditionalExponentialClassifier
from nltk.classify.scikitlearn import SklearnClassifier
print("NLTK libraries imported successfully")

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
print("TensorFlow libraries imported successfully")

# SKLearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix
print("SKLearn libraries imported successfully")

# Other imports
import pandas as pd
import string
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
print("All imports complete!")


# In[3]:


# Download nltk punctuation and stop word data sets
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
class_names = ["sadness","joy","love","anger","fear","suprise"]
# Preprocess text by making lowercase, removing punctuation and stop words.
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words and t not in punctuations]
    return ' '.join(tokens)


# In[5]:


# Read file and set labels
print("Reading File...")
filename = sys.argv[1]
df = pd.read_csv(filename,header=1,names=['index','tweet','emotion_label'])
X = df['tweet']
Y = df['emotion_label']

# Split training and test data
print("Splitting training and test data...")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.1, random_state = 42, stratify=Y)

# Tokenise text, pad to fixed length and apply preprocessing steps
print("Preprocessing text...")

tokeniser = Tokenizer(num_words=10000)

X_train.apply(preprocess_text)
X_test.apply(preprocess_text)

tokeniser.fit_on_texts(X_train)

vocab_size = tokeniser.num_words
max_length = 100

X_train_sequential = tokeniser.texts_to_sequences(X_train)
X_test_sequential = tokeniser.texts_to_sequences(X_test)

X_train_sequential = pad_sequences(X_train_sequential, maxlen=max_length)
X_test_sequential = pad_sequences(X_test_sequential, maxlen=max_length)

print("Text processed!")


# In[6]:


# Building Sequential LSTM model
print("Building model...")
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_sequential, Y_train, epochs=4, batch_size = 256, validation_split=0.2)
print("Model built successfully!")


# In[7]:


# Evaluate model for loss and accuracy based on the test data
loss, accuracy = model.evaluate(X_test_sequential, Y_test, batch_size=256)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


# In[8]:


# Create predictions based on the test data and display in a classification report
predictions = model.predict(X_test_sequential)
predicted_classes = np.argmax(predictions, axis=1)
print(classification_report(Y_test, predicted_classes, target_names=class_names))


# In[12]:


# Create confusion matrix based on the predictions
cm = confusion_matrix(Y_test, predicted_classes)
sns.heatmap(cm, annot=True, fmt="d",cmap="Blues",xticklabels=class_names,yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png",dpi=300)
plt.show()

