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


def read_data(data_path) -> Dict[str, List[str]]:
    dic = {}  # {language_name: [name, ...]}

    for filename in glob.glob(data_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        lines = open(filename, encoding="utf-8").read().strip().split("\n")
        unicodeToAscii: Callable[[str], str] = lambda s: "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn" and c in alphabet)
        lines = [unicodeToAscii(line) for line in lines]
        dic[category] = lines

    return dic


def one_hot_encode(s: str) -> torch.Tensor:
    """
    encode a name from feature set as one-hot tensor:
    put a 1 in the position of the character in the alphabet and 0s elsewhere, then stack them.
    shape: <len(s) x 1 x len(alphabet)>
    """

    tensor = torch.zeros(len(s), 1, len(alphabet))
    for i, c in enumerate(s):
        tensor[i][0][alphabet.index(c)] = 1

    return tensor


def get_random_training_example(data) -> tuple[str, str]:
    category = random.choice(list(data.keys()))
    name = random.choice(data[category])
    return category, name


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
        output = self.softmax(output)  # log_softmax(output)
        return output, hidden

    def init_hidden_input(self):
        return torch.zeros(1, self.hidden_size)


def train_loop(rnn, loss_fn, learning_rate, n_iters):
    pass


def main():
    data: Dict[str, List[str]] = read_data("./data/names/*.txt")  # {language_name: [name, ...]}

    categories = list(data.keys())
    print(f"num languages: {len(categories)=}")  # 18
    print(f"alphabet: {len(alphabet)=}")  # 57

    hyperparams = {
        "hidden_size": 128,
        "learning_rate": 0.005,
    }

    rnn = RNN(input_size=len(alphabet), hidden_size=hyperparams["hidden_size"], output_size=len(categories))
    loss_fn = nn.NLLLoss()  # negative log likelihood loss, since the last layer is log_softmax


if __name__ == "__main__":
    main()
