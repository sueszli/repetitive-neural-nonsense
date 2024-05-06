import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device: {device}")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10, bias=True),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # get prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()  # compute gradients for each parameter
        optimizer.step()  # update weights based on gradients
        optimizer.zero_grad()  # reset gradients so they don't accumulate

        if batch % 100 == 0:
            loss = loss.item()
            current = (batch + 1) * len(X)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    # compute loss and accuracy over entire test set
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # average
    test_loss /= num_batches
    correct /= size

    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    hyperparams = {
        "batch_size": 64,
        "epochs": 3,
        "learning_rate": 1e-3,
    }
    model = NeuralNetwork().to(device)

    train_data = datasets.FashionMNIST(root="./pytorch_data", train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root="./pytorch_data", train=False, download=True, transform=ToTensor())

    train_batcher = DataLoader(train_data, batch_size=hyperparams["batch_size"], shuffle=True)
    test_batcher = DataLoader(test_data, batch_size=hyperparams["batch_size"], shuffle=False)

    loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: SGD = torch.optim.SGD(model.parameters(), lr=hyperparams["learning_rate"])

    for t in range(hyperparams["epochs"]):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_batcher, model, loss_fn, optimizer)
        test_loop(test_batcher, model, loss_fn)


if __name__ == "__main__":
    main()
