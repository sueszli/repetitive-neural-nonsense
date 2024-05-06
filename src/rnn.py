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


alphabet = string.ascii_letters + " .,;'"


def get_data_dict(data_path) -> Dict[str, List[str]]:
    dic = {}  # {language_name: [name, ...]}

    for filename in glob.glob(data_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        lines = open(filename, encoding="utf-8").read().strip().split("\n")
        unicodeToAscii: Callable[[str], str] = lambda s: "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn" and c in alphabet)
        lines = [unicodeToAscii(line) for line in lines]
        dic[category] = lines

    return dic


def one_hot_encode_str(s: str) -> torch.Tensor:
    # put a 1 in the position of the character in the alphabet and 0s elsewhere, then stack them.
    # shape: <len(s) x 1 x len(alphabet)>

    tensor = torch.zeros(len(s), 1, len(alphabet))
    for i, c in enumerate(s):
        tensor[i][0][alphabet.index(c)] = 1

    return tensor


def print_encoded_tensor(tensor: torch.Tensor):
    word = ""
    for i in range(len(tensor)):
        argmax_idx: int = torch.argmax(tensor[i][0]).item()  # type: ignore
        word += alphabet[argmax_idx]
    print("decoded word: ", word)

    for i in range(len(tensor)):
        argmax_idx: int = torch.argmax(tensor[i][0]).item()  # type: ignore
        print(f"i={i} [{alphabet[argmax_idx]}] - ", end=" ")
        for j in range(len(alphabet)):
            content = str(tensor[i][0][j].item())
            content = content.replace("1.0", "\033[92m1\033[0m")
            content = content.replace("0.0", "0")
            print(content, end=" ")
        print()


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


def main():
    data: dict = get_data_dict("./data/names/*.txt")

    categories = list(data.keys())
    print(f"num languages: {len(categories)=}")  # 18
    print(f"alphabet: {len(alphabet)=}")  # 57

    hyperparams = {
        "hidden_size": 128,
        "learning_rate": 0.005,
    }

    rnn = RNN(input_size=len(alphabet), hidden_size=hyperparams["hidden_size"], output_size=len(categories))


if __name__ == "__main__":
    main()
