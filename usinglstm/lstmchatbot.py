import numpy as np
import pickle
from tensorflow import keras
from keras.utils import pad_sequences
import json

# Load the saved model, tokenizer, and label encoder
with open('chatbot_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Define the maximum sequence length
max_sequence_len = model.input_shape[1]

# Define a function to preprocess user input
def preprocess_input(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_len, padding='post')
    return padded_sequence

# Define a function to generate a response
def generate_response(text):
    preprocessed_input = preprocess_input(text)
    predictions = model.predict(preprocessed_input, verbose=0)
    predicted_class_index = np.argmax(predictions)
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
    response = responses[predicted_class_index]
    return response

# Load the responses from the intents dataset
with open('intents.json') as file:
    intents = json.load(file)

responses = []
for intent in intents['intents']:
    responses.append(intent['responses'][0])

# Chatbot interaction loop
while True:
    user_input = input("User: ")
    response = generate_response(user_input)
    print("Chatbot:", response)
