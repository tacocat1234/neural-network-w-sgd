#pragma once

#include <unordered_map>
#include <initializer_list>

using ActivationPair = std::pair<double(*)(double), double(*)(double)>;
using Layer = std::vector<Node>;

class Node{
    public:
        Node(Layer* incomingNodes, std::vector<double> incomingWeights, double bias, ActivationPair activations);
        Node(std::unordered_map<Node*, double> incomingNodesWeights, double bias, ActivationPair activations);

        double getValue();
        double getWeightedSum();
        double setValue();
        double getWeights();
        double getBias();
        double calculateValue();
        double updateBias(double modifier);
        double updateWeight(double modifier, double key);
        std::unordered_map<Node*, double> updateWeights(std::vector<double> modifiers); //gradients are a vector (layers) of a vector (nodes) of map<node, double> (previous layer nodes)
    private:
        double value, preActivation, bias;
        std::unordered_map<Node*, double> incomingNodesWeights;
        UnaryFunctionDouble activationFunction;
};