# Sixian's First Small Neural Network, Version 2
This is a simple implementation of a neural network with SGD, coded from scratch in C++. 

This project was a passion project created during my 9th grade as an introduction to AI and machine learning concepts. It showcases the fundamentals of neural networks and backpropagation, but it is not optimized for performance or scalability.

## Features  

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
Define a custom loss function and its derivative to compute gradients during training. Prebuilt MSE and binary cross entropy functions and derivatives are also implemented.  

## Building  
Requires a C++17 compiler. ppo.cpp is still a work in progress and is not needed.

main.cpp is a demo, it trains a small network on XOR, trains a 784-32-16-10 network on handwritten digits (MNIST) whose layers use different activation functions, and saves its parameters. It then loads them into a new network and tests it without training.  

MSVC (from a Developer Command Prompt for Visual Studio):  
```
cl /O2 /std:c++17 /EHsc main.cpp functions.cpp nodes.cpp neuralNetwork.cpp mnist.cpp /Fe:nn.exe
```

GCC / Clang:  
```
g++ -O2 -std=c++17 main.cpp functions.cpp nodes.cpp neuralNetwork.cpp mnist.cpp -o nn
```

Keep optimization on (/O2 or -O2); without it, MNIST training is much slower.  

## Running  
Demo  
Run from the repository root so the default data path data/mnist resolves:  
```
nn.exe
```
To use MNIST files in a different folder, pass the folder as the first argument:  
```
nn.exe path/to/mnist
```
The demo has three runs, and prints how many seconds each one took.  

## Credit
The MNIST data in data/mnist/ was downloaded from https://github.com/cvdfoundation/mnist.

## Troubleshooting: "Part of this app has been blocked" on Windows  
Sometimes when running demo on Windows, this issue occurs:  

![Windows Security notification: Part of this app has been blocked](images/app_blocked.png)

This is a problem with Windows security feature. Smart App Control blocks it because nn.exe isn't digitally signed. According to the MS support page: 

![Microsoft support page: Smart App Control has blocked part of this app](images/smart_app_control.png)

Workaround: From what I can tell, just wait a few minutes then recompile and run. More permanent fix would be to sign nn.exe with a trusted code signing certificate, but haven't researched how to do that.

## Usage  
Creating a Neural Network  
To create a new instance of the neural network, instantiate an object of the NeuralNetwork class. The constructor parameters are:  

std::initializer_list<size_t> layerSizes: Sizes of each layer in the network (e.g., {2, 3, 2}).  
std::initializer_list<ActivationPair> layerActivations: Activation function pairs ({activation, activationDerivative}) for each layer except the input layer. The input layer is still needed as placeholder.
LossFunction lossFunction: Function pointer for the loss function.  
LossDerivative lossDerivative: Function pointer for the derivative of the loss function.  
double learningRate: A hyperparameter determining the step size during gradient descent.  

--- example---
```c++
NeuralNetwork net(  
    {2, 3, 2},  
    {{sigmoid, sigmoidDerivative}, {tanh, tanhDerivative}, {relu, reluDerivative}},  
    lossMSE,
    lossMSEderivative,  
    0.01  
);  
```

Training  
The train method trains the neural network on the provided dataset:    

---example---
```c++
neuralNetwork.train(TrainingData data, int batchSize = -1);  
```  
TrainingData: A collection of input-output pairs (std::unordered_map<std::vector<double>, std::vector<double>>).  
batchSize: Determines whether to use SGD (1), full-batch gradient descent (data.size()), or mini-batch gradient descent (any value in between). Defaults to -1 (full-batch).

Evaluation  
Use the evaluate function to generate predictions from input data:  

---example---
```c++
neuralNetwork.evaluate(std::vector<double> inputs);  
std::vector<double> result = neuralNetwork.output();
```

Saving and Loading Parameters
downloadParameters() always writes to parameters.txt in the current direcctory.
Values are saved wit the default 6-digit precision.
uploadParameters() needs the activation pairs, because the file only holds sizes, biases and weights.

## NeuralNetwork Class

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

## Future Improvements
If I were to revisit this project, some potential improvements I would consider include:  
Implement techniques like RMSProp or Adam for faster convergence and better optimization.  
Support for L1/L2 regularization to prevent overfitting.  