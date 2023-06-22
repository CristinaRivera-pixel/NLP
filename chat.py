import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Gilbert"
print(f"Hi my name is {bot_name} Let's chat! (type 'quit' to exit)")

name =input('What is your name? (just your name please) ')
print(f"Hi {name}, tell me what you like or dislike and Ill write it down or we can talk about a limited amount of topics")

dislikes = []
likes = []

while True:
    # sentence = "do you use credit cards?"
    sentence = input(f"{name}: ")
    if sentence == "quit":
        break

    rawText = sentence
    sentence = tokenize(sentence)

    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if tag == "User likes":
        print("or here")
        likes.append(rawText)
    if tag == "User dislikes":
        print("here")
        dislikes.append(rawText)
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")


with open(f'{name}.txt', 'w') as f:
    f.write(f"Username:{name}\n")
    f.write("likes:")
    for like in likes:
        f.write("%s\n" % like)
    f.write("dislikes:")
    for dislike in dislikes:
        f.write("%s\n" % dislike)