import time
import onnxruntime as ort
import numpy as np

def test(session, data, target):
    output = session.run([], {"input": data})[0]
    pred = output.argmax(axis=1, keepdims=True)
    correct = np.sum(pred.flatten() == target)
    print(f"Accuracy: {correct}/{1000} ({100.*correct/1000:.0f}%)")

def main():
    # Load data
    npzfile = np.load("data/mnist_test_1000.npz")
    data, target = npzfile["data"], npzfile["target"]

    # Load model
    session = ort.InferenceSession("models/mnist_cnn.onnx")

    # Benchark inference
    start = time.time()
    test(session, data, target)
    end = time.time()
    print(f"Time: {end - start:.4f} seconds")

if __name__ == '__main__':
    main()
