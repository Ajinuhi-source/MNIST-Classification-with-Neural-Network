This repository contains the code for my project on classifying the MNIST dataset using a Neural Network. I developed this project to demonstrate my understanding of neural networks and their application in image classification tasks.

Dataset
The MNIST dataset is a popular dataset in the field of machine learning. It consists of a large collection of grayscale images representing handwritten digits from 0 to 9. The dataset is divided into a training set and a test set, each containing labeled images.

Model Architecture
For this project, I implemented a neural network model using Python and popular deep learning libraries PyTorch. The model architecture consists of input and output layers, as well as one or more hidden layers.

To classify the MNIST images, I designed a neural network with multiple fully connected (dense) layers. The number of neurons and the number of hidden layers can be adjusted based on the complexity of the problem and desired performance.

Training and Evaluation
To train the model, I used the training set from the MNIST dataset. During training, the neural network learns to recognize patterns and features in the images and make predictions accordingly. The model is trained using stochastic gradient descent optimization and a categorical cross-entropy loss function.

After training the model, I evaluated its performance using the test set. The accuracy metric was used to measure how well the model generalizes to unseen data. The trained model achieved 95.92% accuracy on the test set, showcasing its effectiveness in classifying the handwritten digits.
