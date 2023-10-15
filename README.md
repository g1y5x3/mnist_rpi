# Deep Learning Model Inference on MNIST using Raspberry PI 4
This repository provides examples to perform deep learning model inference on MNIST testing dataset (10,000 images) with a few different frameworks that's available on Raspberry Pi 4.

## Training
A basic [CNN model](https://github.com/pytorch/examples/tree/main/mnist) implemented with Pytorch was trained on a PC with Nvidia 3090.
```text
Train Epoch: 14 [59520/60000 (99%)]     Loss: 0.001521

Test set: Average loss: 0.0275, Accuracy: 9909/10000 (99%)
```

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
A sample model weights is provided in **mnist_cnn.pt**, you can run the following to perform inference using **pytorch**. (For installation of pytorch on a raspberry pi 4, you can follow this blog post)

```console
python eval_torch.py
```
Here is a sample output
```text
cpu: # of threads 4
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Net                                      [1000, 10]                --
├─Conv2d: 1-1                            [1000, 32, 26, 26]        320
├─Conv2d: 1-2                            [1000, 64, 24, 24]        18,496
├─Dropout: 1-3                           [1000, 64, 12, 12]        --
├─Linear: 1-4                            [1000, 128]               1,179,776
├─Dropout: 1-5                           [1000, 128]               --
├─Linear: 1-6                            [1000, 10]                1,290
==========================================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
Total mult-adds (G): 12.05
==========================================================================================
Input size (MB): 3.14
Forward/backward pass size (MB): 469.07
Params size (MB): 4.80
Estimated Total Size (MB): 477.01
==========================================================================================

Test set: Average loss: 0.0251, Accuracy: 9915/10000 (99%)

30.0421 seconds
```
