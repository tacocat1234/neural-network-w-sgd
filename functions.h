#pragma once

#include <vector>
#include <stdexcept>
#include <initializer_list>

const double lReluAlpha = 0.1;

//activation functions

using std::tanh;  //tanh activation

double sigmoid(double input);

double relu(double input);

double leakyRelu(double input);

//activation derivatives

double sigmoidDerivative(double input);

double tanhDerivative(double input);

double reluDerivative(double input);

double leakyReluDerivative(double input);


//loss functions

double lossMSE(std::vector<double> actual, std::vector<double> expected); //loss for regression

double lossBinaryCrossEntropy(std::vector<double> actual, std::initializer_list<double> expected); //loss for binary categorization (assuming sigmoid)

std::vector<double> lossMSEderivative(std::vector<double> actual, std::vector<double> expected);

std::vector<double> lossBinaryCrossEntropyDerivative(std::vector<double> actual, std::vector<double> expected);