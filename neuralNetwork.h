#pragma once

#include "nodes.h"
#include <vector>
#include <intializer_list>
#include <pair>
#include <random>
#include <cmath>
#include <iterator>
#include <unordered_map>

using Layer = std::vector<Node>;
using TrainingData = std::unordered_map<std::vector<double>, std::vector<double>>;
using TrainingSample = std::pair<std::vector<double>, std::vector<double>>;
using UnaryFunctionDouble = double(*)(double)
using ActivationPair = std::pair<UnaryFunctionDouble, UnaryFunctionDouble>;
using LossFunction = double(*)(std::vector<double>, std::vector<double>);
using std::size_t;

class NeuralNetwork{
    public:
        NeuralNetwork(std::initializer_list<size_t> layerSizes, std::initializer_list<ActivationPair> layerActivations, LossFunction lossFunction, LossFunction lossDerivative, learningRate = 0.01);

        std::vector<size_t> getSize();
        std::vector<double> output();

        void updateValues();
        void passInputs(std::initializer_list<double> inputs);
        void evaluate(std::initializer_list<double> inputs);
        void evaluate(std::vector<double> inputs);
        
        

        void train(TrainingData data, int batchSize = -1);

    private:
        LossFunction lossFunction;
        LossFunctionlossDerivative;
        std::vector<UnaryFunctionDouble> layerActivationDerivatives;
        void backPropogate(TrainingSample data);
        std::vector<Layer> nodes;
        std::vector<size_t> layerSizes;
        double learningRate;
}