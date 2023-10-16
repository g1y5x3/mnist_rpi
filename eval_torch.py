import time
import torch
import numpy as np
from torchinfo import summary

from train_torch import Net

def test(model, device, data, target):
    model.eval()
    correct = 0
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"Accuracy: {correct}/{1000} ({100.*correct/1000:.0f}%)")


def main():
    device = "cpu"

    # Load data
    npzfile = np.load("data/mnist_test_1000.npz")
    data = torch.tensor(npzfile["data"])
    target = torch.tensor(npzfile["target"])

    # Load model
    model = Net()
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location="cpu"))
    summary(model, input_size=(1000, 1, 28, 28))

    model.to(device)
    model = torch.jit.script(model)

    start = time.time()
    test(model, device, data, target)
    end = time.time()
    print(f"Time: {end - start:.4f} seconds")

if __name__ == '__main__':
    main()
