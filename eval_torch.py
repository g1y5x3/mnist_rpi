import time
import torch
import numpy as np

from train_torch import Net

def test(model, device, data, target):
    model.eval()
    with torch.no_grad():
        data, target = data.to(device), target.to(device)

        start = time.time()
        output = model(data)
        end = time.time()
        print(f"Time: {end - start:.4f} seconds")

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()

    print(f"Accuracy: {correct}/{1000} ({100.*correct/1000:.0f}%)")


def main():
    device = "cpu"

    # Load data
    npzfile = np.load("data/mnist_test_1000.npz")
    data, target = torch.tensor(npzfile["data"]), torch.tensor(npzfile["target"])

    # Load model
    model = Net()
    model.load_state_dict(torch.load("models/mnist_cnn.pt", map_location="cpu"))

    model.to(device)
    model = torch.jit.script(model)

    test(model, device, data, target)

if __name__ == '__main__':
    main()
