import torch
from pathlib import Path
from torchinfo import summary

from train_torch import Net

def main():
    # Load the pytorch model
    model = Net()
    assert Path("mnist_cnn.pt").is_file(), "Did not find pre-trained weights. Run python train_torch.py for training the model."
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location="cpu"))
    model.eval()
    summary(model)

    # Export the model as ONNX
    torch_input = torch.randn(1,1,28,28)
    torch.onnx.export(model, torch_input, "mnist_cnn.onnx", export_params=True, input_names=["input"], output_names=["output"])
    print("exported model to mnist_cnn.onnx")

if __name__ == '__main__':
    main()