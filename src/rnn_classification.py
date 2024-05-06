import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from io import open
import glob
import os
import unicodedata
import string
from typing import Dict, List, Callable
import random
import time
import math


alphabet: str = string.ascii_letters + " .,;'"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def read_data() -> Dict[str, List[str]]:
    dic = {}  # {language_name: [name, ...]}
    data_path: str = "./data/names/*.txt"

    for filename in glob.glob(data_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        lines = open(filename, encoding="utf-8").read().strip().split("\n")
        unicodeToAscii: Callable[[str], str] = lambda s: "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn" and c in alphabet)
        lines = [unicodeToAscii(line) for line in lines]
        dic[category] = lines

    return dic


def one_hot_encode(s: str) -> torch.Tensor:
    """
    encode names into one-hot tensor: put a 1 in the position of the character in the alphabet and 0s elsewhere, then stack them.
    shape: <len(s) x 1 x len(alphabet)>
    """
    tensor = torch.zeros(len(s), 1, len(alphabet))
    for i, c in enumerate(s):
        tensor[i][0][alphabet.index(c)] = 1

    return tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)  # input to hidden
        self.h2h = nn.Linear(hidden_size, hidden_size)  # hidden to hidden
        self.h2o = nn.Linear(hidden_size, output_size)  # hidden to output

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))  # tanh(linear(input) + linear(hidden))

        output = self.h2o(hidden)  # linear(hidden)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


def train_loop(data, rnn, loss_fn, learning_rate, n_iters):
    current_loss = 0
    all_losses = []

    def get_train_example(data):
        category = random.choice(list(data.keys()))
        line = random.choice(data[category])
        category_tensor = torch.tensor([list(data.keys()).index(category)], dtype=torch.long)  # just a number
        line_tensor = one_hot_encode(line)  # <len(line) x 1 x len(alphabet)>
        return category, line, category_tensor, line_tensor

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = get_train_example(data)

        # unravel the RNN
        output = None
        hidden = rnn.init_hidden()
        rnn.zero_grad()
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        assert output is not None

        # compute loss
        loss = loss_fn(output, category_tensor)
        loss.backward()
        current_loss += loss.item()

        # use gradients to update parameters element-wise
        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)  # type: ignore

        # log progress at random intervals
        print_every = 5000
        if iter % print_every == 0:

            # decode label with max probability
            max_prob_idx = output.topk(1)[1][0].item()
            guess: str = list(data.keys())[max_prob_idx]

            print(f"{iter / n_iters * 100:.0f}% | loss: {loss:.4f} | ({line}) -> {guess} | {'true' if guess == category else 'false ' + '(' +  category + ')'}")

        # averages the loss
        plot_every = 1000
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    return all_losses


def evaluate(line_tensor, rnn):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output


def predict(input_line, data, rnn, n_predictions=3):
    print("\n> %s" % input_line)
    with torch.no_grad():
        output = evaluate(one_hot_encode(input_line), rnn)

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            all_categories = list(data.keys())
            print("(%.2f) %s" % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])


def main():
    hyperparams = {
        "hidden_size": 128,
        "learning_rate": 0.005,  # very sensitive
        "n_iters": 100000,
    }

    data = read_data()
    rnn = RNN(len(alphabet), hyperparams["hidden_size"], len(data.keys()))
    loss_fn = nn.NLLLoss()  # negative log likelihood loss - since the last layer is log softmax
    all_losses = train_loop(data, rnn, loss_fn, hyperparams["learning_rate"], hyperparams["n_iters"])

    predict("Dovesky", data, rnn)


if __name__ == "__main__":
    main()
