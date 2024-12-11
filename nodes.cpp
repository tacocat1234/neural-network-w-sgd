#include "nodes.h"

Node::Node(Layer* incomingNodes, std::vector<double> incomingWeights, double bias, UnaryFunctionDouble activation) : bias(bias), activationFunction(activation){
    if (incomingNodes.size() != incomingWeights.size()){
        throw std::invalid_argument("Incoming Nodes and Weights size are not equal.")
    }
    for (int i = 0; i < (*incomingNodes).size(); i++){
        incomingNodesWeights[&((*incomingNodes)[i])] = incomingWeights[i];
    }
}

Node::Node(std::unordered_map<Node*, double> incomingNodesWeights, double bias, UnaryFunctionDouble activation) : incomingNodesWeights(incomingNodesWeights), bias(bias), activationFunction(activation) {}

double Node::getValue(){
    return value;
}

double getWeightedSum(){
    return preActivation;
}

double Node::setValue(double newVal){
    value = newVal;
    return value;
}
double Node::getWeights(){
    return incomingNodesWeights;
}
double Node::getBias(){
    return bias;
}

double Node::calculateValue(){
    value = 0;
    for (const auto& pair : incomingNodesWeights){
        value += pair.first.getValue() * pair.second;
    }
    value += bias;
    preActivation = value;
    value = activationFunction(value);
    return value;
}

double updateBias(double modifier){
    bias += modifier;
    return bias;
}

double updateWeight(double modifier, double key){
    incomingNodesWeights[key] -= modifier;
    return incomingNodesWeights[key];
}

std::unordered_map<Node*, double> updateWeights(std::vector<double> modifiers){
    if (modifiers.size() != incomingNodesWeights.size()){
        throw std::invalid_argument("Update weights count not equal to number of weights present");
    }

    for (int i = 0; i < modifiers.size(); i++){
        incomingNodesWeights[i].second += modifiers[i];
    }
    return incomingNodesWeights;
}