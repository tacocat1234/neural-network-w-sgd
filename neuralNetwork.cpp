#include "neuralNetwork.h"
#include <random>
#include <cmath>

using Weights = std::vector<double>;
using LayerWeights = std::vector<Weights>;
using LayerValues = std::vector<double>;

NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layerSizes,  std::initializer_list<ActivationPair> layerActivations, LossFunction lossFunction, LossDerivative lossDerivative, double learningRate)
    : layerSizes(layerSizes), lossFunction(lossFunction), lossDerivative(lossDerivative), learningRate(learningRate) {
    // Random number generator for Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (const auto& pair : layerActivations){
        this->layerActivationDerivatives.emplace_back(pair.second);
    }

    Layer* prevLayer;
    auto activationPtr = layerActivations.begin();
    for (size_t layerSize : layerSizes) {
        Layer layer;

        // Determine number of incoming connections
        size_t numIncoming = prevLayer->size(); // Previous layer size

        for (size_t i = 0; i < layerSize; ++i) {
            // Generate weights using Xavier initialization
            std::vector<double> weights;
            double bias = 0.0;
            
            if (numIncoming > 0) {
                double xavierLimit = std::sqrt(6.0 / (numIncoming + layerSize));
                std::uniform_real_distribution<double> dist(-xavierLimit, xavierLimit);
                
                for (size_t j = 0; j < numIncoming; ++j) {
                    weights.push_back(dist(gen)); // Random weight
                }

                bias = dist(gen);
            } else { //is input layer
                weights.resize(numIncoming, 0.0);
                bias = 0.0;
            }
            
            layer.emplace_back(Node(prevLayer, weights, bias, (*activationPtr).first));  //correct
        }

        nodes.emplace_back(std::move(layer));
        prevLayer = &nodes.back();
        activationPtr++;
    }
}

std::vector<size_t> NeuralNetwork::getSize(){
    return layerSizes;
}

std::vector<double> NeuralNetwork::output(){
    return nodes[nodes.size() - 1];
}

void NeuralNetwork::updateValues(){
    for (auto& layer : nodes){
        for (auto& node : layer){
            node.calculateValue();
        }
    }
}

void NeuralNetwork::passInputs(std::initializer_list<double> inputs){
    if (inputs.size() != nodes[0].size()){
        throw std::invalid_argument("The size of inputs and the amount of input nodes must be the same");
    }
    for (int i = 0; i < inputs.size(); i++){
        nodes[0][i].setValue(inputs[i]);
    }
}

void NeuralNetwork::evaluate(std::vector<double> inputs){
    passInputs(inputs);
    updateValues();
}

void NeuralNetwork::evaluate(std::initializer_list<double> inputs){
    evaluate()
}

void NeuralNetwork::backPropogate(TrainingSample data){
    LayerValues inputs = data.first;
    LayerValues expectedOut = data.second;

    evaluate(inputs);

    std::vector<LayerValues> preActGradients; //temp, for ease of computing, equivalent to bias
    double curWeightGradient

    std::vector<double> lossGradients = lossDerivative(output(), expectedOut);

    preActGradients.resize(layerSizes.size(), std::vector<double>{});

    for (int i = layerSizes.size() - 1; i >= 0; i--){ //for each layer, backwards

        preActGradients[i].resize(layerSizes[i], 0);

        for (int j = 0; j < layerSizes[i]; j++){ //for each node in layer

            if (i != layerSizes.size() - 1) {
                for (int k = 0; k < layerSizes[i + 1]; k++) {//for each in next layer
                    preActGradients[i][j] += preActGradients[i + 1][k] * nodes[i + 1][k].getWeights()[&(nodes[i][j])]; //multiply prevDv * weight i+1, k to i, j, or prevDv * ∂nextPreAct/∂curPostAct
                }//            ___  * ∂a_n/∂z_n (activationDerivative(z))
                preActGradients[i][j] *= layerActivationDerivatives[i](nodes[i][j].getWeightedSum()); //multiply by ∂a_n/∂preAct_n
            } else {
                preActGradients[i][j] = lossGradients[i] * layerActivationDerivatives[i](nodes[i][j].getWeightedSum()); //partial derivative of loss w/resp. preActivation val of node in output layer1`
            }
            nodes[i][j].updateBias(preActGradients[i][j] * learningRate);

            if (i > 0){
                for (int k = 0; k < layerSizes[i - 1]; k++){ //for each node in incomingLayer, i.e. for each weight to the node
                    curWeightGradient = preActGradients[i][j] * nodes[i - 1][k].getValue();
                    nodes[i][j].updateWeight(curWeightGradient * learningRate, &(nodes[i - 1][k])); //update weight from i-1,k to i,j by weightGradients * learnignRate
                }
            }
        }
    }
}//bias gradients are equivalent to the preActivation gradients at any node n