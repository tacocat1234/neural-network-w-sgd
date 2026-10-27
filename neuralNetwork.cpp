#include "neuralNetwork.h"

#include <iostream>


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

    Layer inputLayer;
    Layer* prevLayer = &inputLayer;
    auto activationPtr = layerActivations.begin();
    for (size_t layerSize : layerSizes) {
        Layer layer;

        // Determine number of incoming connections
        size_t numIncoming = prevLayer->size(); // Previous layer size

        for (size_t i = 0; i < layerSize; i++) {
            // Generate weights using Xavier initialization
            std::vector<double> weights;
            double bias = 0.0;
            
            if (numIncoming > 0) {
                double xavierLimit = std::sqrt(6.0 / (numIncoming + layerSize));
                std::uniform_real_distribution<double> dist(-xavierLimit, xavierLimit);
                
                for (size_t j = 0; j < numIncoming; j++) {
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
    std::vector<double> output;
    size_t i = 0;
    for (auto& node : nodes[nodes.size() - 1]){
        output.push_back(node.getValue());
        i++;
    }
    return output;
}

void NeuralNetwork::updateValues(){
    // skip layer 0, the input layer
    for (int i = 1; i < nodes.size(); i++){
        for (auto& node : nodes[i]){
            node.calculateValue();
        }
    }
}

void NeuralNetwork::passInputs(const std::vector<double>& inputs){
    if (inputs.size() != nodes[0].size()){
        throw std::invalid_argument("The size of inputs and the amount of input nodes must be the same");
    }
    for (int i = 0; i < inputs.size(); i++){
        nodes[0][i].setValue(inputs[i]);
    }
}

void NeuralNetwork::passInputs(std::initializer_list<double> inputs){
    passInputs(std::vector<double>(inputs));
}

void NeuralNetwork::evaluate(const std::vector<double>& inputs){
    passInputs(inputs);
    updateValues();
}

void NeuralNetwork::evaluate(std::initializer_list<double> inputs){
    evaluate(std::vector<double>(inputs));
}

void NeuralNetwork::backPropagate(const TrainingSample& data){
    if (layerSizes.size() == 1){
        return;
    }

    const LayerValues& inputs = data.first;
    const LayerValues& expectedOut = data.second;

    evaluate(inputs);

    LayerValues prevWSumGradients; //previous weighted sum gradients
    LayerValues currentWSumGradients = std::vector<double>(layerSizes[layerSizes.size() - 1], 0.0); //current weighted sum gradients
    double curWeightGradient;

    std::vector<double> lossGradients = lossDerivative(output(), expectedOut);

    for (int i = layerSizes.size() - 1; i > 0; i--){ //for each layer, backward

        for (int j = 0; j < layerSizes[i]; j++){ //for each node in layer

            if (i != layerSizes.size() - 1) {
                for (int k = 0; k < layerSizes[i + 1]; k++) {//for each in next layer
                    currentWSumGradients[j] += prevWSumGradients[k] * nodes[i + 1][k].getWeights().at(&(nodes[i][j]));
                }//            ___  * ∂a_n/∂z_n (activationDerivative(z))
                currentWSumGradients[j] *= layerActivationDerivatives[i](nodes[i][j].getWeightedSum()); //multiply by ∂a_n/∂preAct_n
            } else {
                currentWSumGradients[j] = lossGradients[j] * layerActivationDerivatives[i](nodes[i][j].getWeightedSum()); //partial derivative of loss w/resp. preActivation val of node in output layer1`
            }
            nodes[i][j].updateBias(currentWSumGradients[j] * learningRate);

            if (i > 0){
                for (int k = 0; k < layerSizes[i - 1]; k++){ //for each node in incomingLayer, i.e. for each weight to the node
                    curWeightGradient = currentWSumGradients[j] * nodes[i - 1][k].getValue(); //derivative w/respect current node z * derivative of curNode z w/r w from i-1,k 
                    nodes[i][j].updateWeight(curWeightGradient * learningRate, &(nodes[i - 1][k])); //update weight from i-1,k to i,j by weightGradients * learnignRate
                }
            }
        }
        prevWSumGradients = currentWSumGradients;
        currentWSumGradients = std::vector<double>(layerSizes[i - 1], 0.0); //clear, set size to next layer in loop (prev layer in network)
    }
}//bias gradients are equivalent to the preActivation gradients at any node n

void NeuralNetwork::train(const TrainingData& data, int batchSize){
    if (batchSize <= 0 || batchSize > data.size()) {
        batchSize = data.size(); // Use all data if batchSize is invalid
    }
    //pointers into data so shuffling doesn't copy the samples
    std::vector<const TrainingData::value_type*> samples;
    samples.reserve(data.size());
    for (const auto& sample : data){
        samples.push_back(&sample);
    }

    std::random_device rd; //shuffle
    std::mt19937 gen(rd());
    std::shuffle(samples.begin(), samples.end(), gen);

    //iterate over batches
    for (std::size_t i = 0; i < samples.size(); i += batchSize){
        auto batchEnd = std::min(i + batchSize, samples.size()); //makes sure end of batch is within bounds of data

        for (std::size_t s = i; s < batchEnd; s++) {//for each sample in batch
            backPropagate(*samples[s]);
        }
    }
}

void NeuralNetwork::downloadParameters(){
    std::ofstream outFile("parameters.txt");
    if (outFile.is_open()){
        outFile << nodes.size() << "\n"; //write number of layers
        for (auto& each : nodes){
            outFile << each.size() << "\n"; //write number of nodes per layer
        } 
        for (int i = 0; i < nodes.size(); i++){
            for (auto& node : nodes[i]){
                outFile << node.getBias() << " ";
                if (i > 0){
                    const auto& weights = node.getWeights();
                    for (auto& prevNode : nodes[i - 1]){
                        outFile << weights.at(&prevNode) << " ";
                    }
                }
                outFile << "\n";
            }
        }
        outFile.close();
    } else{
        std::cerr << "Unable to open file for writing" << std::endl;
    }
}

void NeuralNetwork::uploadParameters(const std::string& fileName, std::initializer_list<ActivationPair> layerActivations){
    std::ifstream inFile(fileName);
    if (inFile.is_open()){
        this->layerActivationDerivatives.clear();
        this->nodes.clear();
        for (const auto& pair : layerActivations){
            this->layerActivationDerivatives.emplace_back(pair.second);
        }

        int layerCount;
        inFile >> layerCount;
        std::vector<int> layerSizes;
        layerSizes.resize(layerCount);
        for (int i = 0; i < layerCount; i++){
            inFile >> layerSizes[i];
        } //read all layerSizes

        Layer inputLayer;
        Layer* prevLayer = &inputLayer;
        this->nodes.reserve(layerSizes.size());
        auto activationPtr = layerActivations.begin();
        for (size_t layerSize : layerSizes) {
            Layer layer;

            // Determine number of incoming connections
            size_t numIncoming = prevLayer->size(); // Previous layer size

            for (size_t i = 0; i < layerSize; i++) {
                std::vector<double> weights;
                double bias;
                inFile >> bias;
                
                if (numIncoming > 0) {
                    double cWeight;
                    for (size_t j = 0; j < numIncoming; j++) {
                        inFile >> cWeight;
                        weights.push_back(cWeight);
                    }
                }
                layer.emplace_back(Node(prevLayer, weights, bias, (*activationPtr).first));  //correct
            }

            nodes.emplace_back(std::move(layer));
            prevLayer = &nodes.back();
            activationPtr++;
        }
        inFile.close();
    } else{
        std::cerr << "Unable to open file for reading" << std::endl;
    }
}