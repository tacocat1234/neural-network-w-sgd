#pragma once

#include <unordered_map>
#include <initializer_list>
#include <vector>
#include <stdexcept>

class Node;
using UnaryFunctionDouble = double(*)(double);
using ActivationPair = std::pair<UnaryFunctionDouble, UnaryFunctionDouble>;
using Layer = std::vector<Node>;


class Node{
    public:
        Node(Layer* incomingNodes, const std::vector<double>& incomingWeights, double bias, UnaryFunctionDouble activations);
        Node(std::unordered_map<Node*, double>& incomingNodesWeights, double bias, UnaryFunctionDouble activations);

        double getValue();
        double getWeightedSum();
        double setValue(double newVal);
        const std::unordered_map<Node*, double>& getWeights() const;
        double getBias();
        double calculateValue();
        double updateBias(double modifier);
        double updateWeight(double modifier, Node* key);
        const std::unordered_map<Node*, double>& updateWeights(const std::vector<double>& modifiers); //gradients are a vector (layers) of a vector (nodes) of map<node, double> (previous layer nodes)
    private:
        double value, preActivation, bias;
        std::unordered_map<Node*, double> incomingNodesWeights;
        UnaryFunctionDouble activationFunction;
};