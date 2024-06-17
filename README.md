# neural-network-w-sgd
This is a simple implementation of a neural network with SGD, coded from scratch in C++. 

This was a passion project of mine I completed late into my freshman year of high school, and was my entryway into more difficult AI projects, such as LLMs like the MAMBA architecture. Do not expect this to be particularly optimized or fast, this was written by a 9th grader, and is merely a demonstration and exploration of what I can do.

To create new instance of NeuralNetwork, create an object of type NeuralNetwork.
The parameters are as follows: 
vector<double> layerSizes, a list containing the sizes of the layers in the network, {2, 3, 2} would be a network with 2 input nodes, a single layer of 3 hidden nodes, and 2 output nodes
functionPtr activation, a function that takes in a double and outputs a double that acts as the activation. tanh, sigmoid, and relu are built in.
functionPtr activationDerivative, the derivative to the activation function. tanhDerivative, sigmoidDerivative, and reluDerivative are built in.
double learningRate, a hyperparameter deciding how much the neural net jumps per backpropagation. Higher values converge faster but may get stuck more often.
int batchSize, a hyperparameter deciding how many data points are evaluated per batch, 1 being SGD, data.size() being normal GD, and anything inbetween being mini-batch GD.

The relevant functions are:
neuralNetwork.train(int iterations, trainingData& dataSet);
This trains the net using data dataSet using iteration iterations.
trainingData is a type alias for vector<pair<vector<double>, vector<double>>>
This means it is a vector containing pairs of vectors, with each pair mapping to a vector of input values and the corresponding expected vector of output values
The AI does not retain information between runs, I might work on that later but you must retrain it every time as is.

neuralNetwork.evaluate(vector<double> inputs);
This uses the neural net to evaluate the given inputs inputs and returns the net's outputs as a vector of doubles.

If i were to return to this project, I would likely plan on implementing saving weights/biases and RMSPROP next.
