import json
import numpy as np
import pickle
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load the intents dataset
with open('intents.json') as file:
    intents = json.load(file)

# Extract the patterns and responses from the intents
patterns = []
responses = []
tags = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern.lower())  # Convert to lowercase
        responses.append(intent['responses'][0].lower())  # Convert to lowercase
    tags.append(intent['tag'])

# Tokenize the patterns
tokenizer = Tokenizer(lower=True)  # Set 'lower' to True for case insensitivity
tokenizer.fit_on_texts(patterns)
vocab_size = len(tokenizer.word_index) + 1

# Convert patterns to sequences of integers
sequences = tokenizer.texts_to_sequences(patterns)
max_sequence_len = max(len(seq) for seq in sequences)

# Pad sequences to have the same length
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')

# Encode the responses using LabelEncoder
label_encoder = LabelEncoder()
encoded_responses = label_encoder.fit_transform(responses)

# Define the model architecture
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_sequence_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Use the number of classes as the output size

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, encoded_responses, epochs=200, batch_size=16)

# Save the trained model using pickle
with open('chatbot_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the tokenizer and label encoder for future use
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)