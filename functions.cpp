#include "functions.h"
#include <cmath>
#include <set>

//activation functions

double sigmoid(double input){
    return 1.0 / (1.0 + exp(-input));
}

double relu(double input){
    return input > 0 ? input : 0.0;
}

double leakyRelu(double input){
    return (input > 0) ? input : lReluAlpha * input;
}

//activation derivaive

double sigmoidDerivative(double input){
    return sigmoid(input) * (1 - sigmoid(input));
}

double tanhDerivative(double input){
    return 1.0 / (std::cosh(input) * std::cosh(input));
}

double reluDerivative(double input){
    return input >= 0 ? 1 : 0;
}

double leakyReluDerivative(double input){
    return input >= 0 ? 1 : lReluAlpha;
}

//loss functions 

double lossMSE(std::vector<double> actual, std::vector<double> expected){
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("Size mismatch between actual and expected values.");
    }

    double mse = 0.0;
    auto itActual = actual.begin();
    auto itExpected = expected.begin();

    for (; itActual != actual.end(); ++itActual, ++itExpected) {
        double diff = *itActual - *itExpected;
        mse += diff * diff;
    }

    return mse / actual.size();
}

double lossBinaryCrossEntropy(std::vector<double> actual, std::vector<double> expected){
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("Size mismatch between actual and expected values.");
    }

    double bce = 0.0;
    auto itActual = actual.begin();
    auto itExpected = expected.begin();

    for (; itActual != actual.end(); ++itActual, ++itExpected) {
        if (*itActual < 0.0 || *itActual > 1.0 || *itExpected < 0.0 || *itExpected > 1.0) {
            throw std::invalid_argument("Values for Binary Cross-Entropy must be between 0 and 1.");
        }
        bce += (*itExpected * log(*itActual)) + ((1.0 - *itExpected) * log(1.0 - *itActual));
    }

    return -bce / actual.size();
}

std::vector<double> lossMSEderivative(std::vector<double> actual, std::vector<double> expected) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("Vectors 'actual' and 'expected' must have the same size.");
    }

    std::vector<double> derivative(actual.size());
    size_t N = actual.size();
    for (size_t i = 0; i < N; ++i) {
        derivative[i] = 2.0 * (actual[i] - expected[i]) / static_cast<double>(N);
    }
    return derivative;
}

std::vector<double> lossBinaryCrossEntropyDerivative(std::vector<double> actual, std::vector<double> expected) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("Vectors 'actual' and 'expected' must have the same size.");
    }

    std::vector<double> derivative(actual.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] == 0 || actual[i] == 1) {
            throw std::domain_error("Actual values must be strictly between 0 and 1 to avoid division by zero.");
        }
        derivative[i] = -(expected[i] / actual[i]) + ((1.0 - expected[i]) / (1.0 - actual[i]));
    }
    return derivative;
}