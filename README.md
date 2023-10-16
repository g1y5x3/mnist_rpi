# Deep Learning Model Inference on MNIST using Raspberry PI 4
![mnist](figures/mnist.png)

This repository provides examples to perform deep learning model inference on MNIST testing dataset (10,000 images) with a few different frameworks that's available on Raspberry Pi 4.

## Training
A basic [CNN model](https://github.com/pytorch/examples/tree/main/mnist) implemented with Pytorch was trained on a PC with Nvidia 3090.

If you would like to repeat the training, first, you need to create a folder for storing MNIST dataset.
```console
mkdir data
```
then run the following command to train the model and save the weights.
```console
python train.py --save-model 
```

## Inference

### Pytorch (1.13.0)
A sample model weights is provided in `mnist_cnn.pt`, you can run the following to perform inference using **pytorch**. (For installation of pytorch on a raspberry pi 4, you can follow this blog post)

```console
python eval_torch.py
```

### ONNX
To export the **pytorch** model to **ONNX**
```
python torch_to_onnx.py
```
Here is a graph generated with [netron](https://netron.app/) from `mnist_cnn.onnx`
![model_graph](figures/mnist_cnn.onnx.png)