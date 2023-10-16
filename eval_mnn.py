import time
import MNN
import numpy as np

def test(model, data, target):
    input_var = MNN.expr.placeholder([1000,1,28,28], MNN.expr.NCHW)
    input_var.write(data)

    start = time.time()
    output = model.forward([input_var])
    end = time.time()
    print(f"Time: {end - start:.4f} seconds")

    output = output[0]
    pred = np.argmax(output.read(), axis=1, keepdims=True)
    correct = np.sum(pred.flatten() == target)
    print(f"Accuracy: {correct}/{1000} ({100.*correct/1000:.0f}%)")

def main():
    # Load data
    npzfile = np.load("data/mnist_test_1000.npz")
    data, target = npzfile["data"], npzfile["target"]

    # Load model
    model = MNN.nn.load_module_from_file("models/mnist_cnn.mnn", ["input"], ["output"])

    # Benchark inference
    test(model, data, target)

if __name__ == '__main__':
    main()