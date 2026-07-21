from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

# Load model only once when Django starts
model = load_model("chatbot (1).h5")

#Load tokenizer
with open("tokenizer (1).pkl", "rb") as f:
    tokenizer = pickle.load(f)
#
 #Load label encoder
with open("label_encoder (1).pkl", "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = 5


def chatbot(request):
    question=""
    answer = ""

    if request.method == "POST":

        question = request.POST.get("question")

        # tokenizer = Tokenizer()
        # tokenizer.fit_on_texts(question)
        #
        # label_encoder = LabelEncoder()
        # label_encoder.fit([answer])

        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([question])

        # Padding
        padded = pad_sequences(sequence, maxlen=MAX_LEN,padding="post")

        # Prediction
        prediction = model.predict(padded)

        # Highest probability class
        index = np.argmax(prediction)

        # Convert class index to text
        answer = label_encoder.inverse_transform([index])[0]

    #return render(request, "chatbot.html", {"answer": answer})
    return render(request, "chatbot.html", {
        "question": question if request.method == "POST" else "",
        "answer": answer})