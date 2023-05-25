##### importing the required libraries for deploying the model on the local system
from flask import Flask, request
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
app = Flask(__name__)

# Load the saved model
with open("lstm_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def generate_text():
    if request.method == "POST":
        seed_text = request.form["seed_text"]
        num_words = int(request.form["num_words"])

        generated_sentence = generate_sentence(seed_text, num_words)

        return generated_sentence

    return """
        <form method="POST">
            <input type="text" name="seed_text" placeholder="Enter seed text">
            <input type="number" name="num_words" placeholder="Enter the number of words to generate">
            <input type="submit" value="Generate">
        </form>
    """

if __name__ == "__main__":
    app.run()


