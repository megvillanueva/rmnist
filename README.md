#Reduced MNIST (RMNIST)

MNIST ("Modified National Institute of Standards and Technology") is a widely available data set for digit recognition. Each image of digit is written as a 28 x 28 matrix with values 0-255, representing its color in grayscale (0 as black, 255 as white). MNIST has 42,000 images for training and 28,000 images for testing.

With Reduced MNIST or RMNIST, I tried to tackle the problem with a smaller data set, getting only 10,000 images from the original data set.

## Implementation
Read more about the implementation [here.](https://docs.google.com/document/d/127iTGWeFwMHHneBrBLsP6x3m9-XB_Aa9mE1Fliw7NSw/edit?usp=sharing)

### TLDR;
CNN (LeNet-5) with dropout and data augmentation is done. Accuracy achieved is 99%.


## Dependencies
* Python 3.6
* [Keras 2.2.4](https://keras.io/)
* [Tensorflow 1.9.0](https://www.tensorflow.org/)

## Installation
1. Install dependencies above
2. `$ pip install -r requirements.txt`

## Running
```
python train.py DATA_DIR
```