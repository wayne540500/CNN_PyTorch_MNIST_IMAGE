MNIST Classifier using PyTorch
This repository contains a PyTorch implementation of a simple Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset.

Prerequisites:

Python 3.x
PyTorch
Matplotlib
NumPy
Pandas
Scikit-Learn

Installation:

Install the required libraries using pip:

pip install torch torchvision numpy pandas scikit-learn matplotlib

Dataset:

The MNIST dataset, which consists of 70,000 28x28 grayscale images of the ten digits, along with a training set of 60,000 examples, and a test set of 10,000 examples, is used in this project.

Model Description:

The model is a CNN with two convolutional layers, followed by max-pooling layers, and three fully connected layers at the end. The model architecture is as follows:

Convolutional layer with 6 filters
Convolutional layer with 16 filters
Fully connected layer with 120 units
Fully connected layer with 84 units
Output layer with 10 units (for 10 classes)
Training
The model is trained over 5 epochs, using the Adam optimizer and Cross-Entropy Loss function. The performance of the model is evaluated using training and validation accuracy and loss.

Usage:

To train the model, run the Python script with the model and training code. The script will download the MNIST dataset, train the model, and print the training progress. The training process includes both the training and validation phases.

Testing:

After training, the model's performance is evaluated on the test dataset to report the final accuracy.

![image](https://github.com/wayne540500/CNN_PyTorch_MNIST_IMAGE/assets/69573286/281d4c16-21f0-44c7-af25-bd9a69705485)

![image](https://github.com/wayne540500/CNN_PyTorch_MNIST_IMAGE/assets/69573286/1c3eb012-94ca-4697-b8ad-90724339180d)

![image](https://github.com/wayne540500/CNN_PyTorch_MNIST_IMAGE/assets/69573286/8736d38a-1067-4f96-91fa-09c068f666f3)

Visualization:

The loss and accuracy over epochs are plotted using Matplotlib, providing insights into the training process.

Example Prediction:

An example from the test dataset is used to demonstrate a prediction. The script reshapes and displays the image, and then uses the trained model to predict the digit.

Customization:

For testing with different samples, change the index in the test dataset (e.g., change 4143 to another index like 1994 or 2001).

Repository Structure:

model.py: Contains the CNN model definition.
train.py: Script for training the model.
test.py: Script for evaluating the model's performance on the test set.
utils.py: Contains utility functions for data loading and transformation.
