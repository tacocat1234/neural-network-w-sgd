# neural-network-w-sgd
This is a simple implementation of a neural network with SGD, coded from scratch in C++. 

This project was a passion project created during my freshman year of high school as an introduction to AI and machine learning concepts. It showcases the fundamentals of neural networks and backpropagation, but it is not optimized for performance or scalability. This implementation is meant for educational purposes and as a starting point for understanding neural networks.

Features

Customizable Neural Network Architecture
Specify the size and structure of the network using a list of layer sizes.
Example: {2, 3, 2} creates a network with 2 input nodes, one hidden layer with 3 nodes, and 2 output nodes.
Activation Functions

Built-in support for common activation functions: tanh, sigmoid, ReLU, and leakyReLU.
Their derivatives (tanhDerivative, sigmoidDerivative, reluDerivative, leakyReluDerivative) are also included for backpropagation.
You can provide custom activation functions if needed.

Training
Supports stochastic, mini-batch, and full-batch gradient descent by adjusting the batchSize parameter.
Utilizes backpropagation with a customizable learning rate.

Loss Functions
Define a custom loss function and its derivative to compute gradients during training. Prebuilt MSE and binary cross entropy functions and derivatives are also implemented

Usage
Creating a Neural Network
To create a new instance of the neural network, instantiate an object of the NeuralNetwork class. The constructor parameters are:

std::initializer_list<size_t> layerSizes: Sizes of each layer in the network (e.g., {2, 3, 2}).
std::initializer_list<ActivationPair> layerActivations: Activation function pairs ({activation, activationDerivative}) for each layer.
LossFunction lossFunction: Function pointer for the loss function.
LossFunction lossDerivative: Function pointer for the derivative of the loss function.
double learningRate: A hyperparameter determining the step size during gradient descent.

---example---

NeuralNetwork net(
    {2, 3, 2},
    {{sigmoid, sigmoidDerivative}, {tanh, tanhDerivative}, {relu, reluDerivative}},
    meanSquaredError,
    meanSquaredErrorDerivative,
    0.01
);

Training
The train method trains the neural network on the provided dataset:

---example---

neuralNetwork.train(TrainingData data, int batchSize = -1);

TrainingData: A collection of input-output pairs (std::unordered_map<std::vector<double>, std::vector<double>>).
batchSize: Determines whether to use SGD (1), full-batch gradient descent (data.size()), or mini-batch gradient descent (any value in between). Defaults to -1 (full-batch).

Evaluation
Use the evaluate function to generate predictions from input data:

---example---

std::vector<double> result = neuralNetwork.evaluate(std::vector<double> inputs);

NeuralNetwork Class

Public Functions:
getSize: Returns the network's layer sizes.
output: Returns the output of the most recent evaluation.
updateValues: Updates the values of all nodes by forward propagation.
evaluate: Takes an input vector, passes it through the network, and returns the output.
train: Handles training using backpropagation and gradient descent.

Private Functions:
backPropagate: Implements backpropagation to adjust weights and biases based on gradients.

Node Class (in nodes.h)
Each Node represents a single neuron in the network, storing its:

Input connections (weights and biases).
Current value (activation output).
Activation function and its derivative.

If I were to revisit this project, some potential improvements I would consider include:
Add functionality to save weights and biases for reuse without retraining.
Implement techniques like RMSProp or Adam for faster convergence and better optimization.
Support for L1/L2 regularization to prevent overfitting.
