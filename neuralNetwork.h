#pragma once

#include "nodes.h"
#include <vector>
#include <stdexcept>
#include <utility>
#include <random>
#include <cmath>
#include <iterator>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <string>
#include <functional>

using Layer = std::vector<Node>;

// hash for std::vector<double>, needed by TrainingData
// adapted from https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
namespace std {
    template <>
    struct hash<std::vector<double>> {
        size_t operator()(const std::vector<double>& v) const noexcept {
            size_t seed = v.size();
            for (double d : v) {
                seed ^= std::hash<double>()(d) + 0x9e3779b8 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
}
using TrainingData = std::unordered_map<std::vector<double>, std::vector<double>>;
using TrainingSample = std::pair<std::vector<double>, std::vector<double>>;
using UnaryFunctionDouble = double(*)(double);
using ActivationPair = std::pair<UnaryFunctionDouble, UnaryFunctionDouble>;
using LossFunction = double(*)(const std::vector<double>&, const std::vector<double>&);
using LossDerivative = std::vector<double>(*)(const std::vector<double>&, const std::vector<double>&);
using std::size_t;

class NeuralNetwork{
    public:
        NeuralNetwork(std::initializer_list<size_t> layerSizes, std::initializer_list<ActivationPair> layerActivations, LossFunction lossFunction, LossDerivative lossDerivative, double learningRate = 0.01);

        std::vector<size_t> getSize();
        std::vector<double> output();

        void updateValues();
        void passInputs(const std::vector<double>& inputs);
        void passInputs(std::initializer_list<double> inputs);
        void evaluate(std::initializer_list<double> inputs);
        void evaluate(const std::vector<double>& inputs);



        void train(const TrainingData& data, int batchSize = -1);

        void downloadParameters();
        void uploadParameters(const std::string& fileName, std::initializer_list<ActivationPair> layerActivations);

    private:
        LossFunction lossFunction;
        LossDerivative lossDerivative;
        std::vector<UnaryFunctionDouble> layerActivationDerivatives;
        void backPropagate(const TrainingSample& data);
        std::vector<Layer> nodes;
        std::vector<size_t> layerSizes;
        double learningRate;
};