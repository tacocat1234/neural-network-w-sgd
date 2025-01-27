#include "neuralNetwork.h"


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
        output[i] = node.getValue();
        i++;
    }
    return output;
}

void NeuralNetwork::updateValues(){
    for (auto& layer : nodes){
        for (auto& node : layer){
            node.calculateValue();
        }
    }
}

void NeuralNetwork::passInputs(std::vector<double> inputs){
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

void NeuralNetwork::evaluate(std::vector<double> inputs){
    passInputs(inputs);
    updateValues();
}

void NeuralNetwork::evaluate(std::initializer_list<double> inputs){
    evaluate(std::vector<double>(inputs));
}

void NeuralNetwork::backPropogate(TrainingSample data){
    LayerValues inputs = data.first;
    LayerValues expectedOut = data.second;

    evaluate(inputs);

    LayerValues prevWSumGradients; //previous weighted sum gradients
    LayerValues currentWSumGradients = std::vector<double>(layerSizes[layerSizes.size() - 1], 0.0); //current weighted sum gradients
    double curWeightGradient;

    std::vector<double> lossGradients = lossDerivative(output(), expectedOut);

    for (int i = layerSizes.size() - 1; i >= 0; i--){ //for each layer, backward

        for (int j = 0; j < layerSizes[i]; j++){ //for each node in layer

            if (i != layerSizes.size() - 1) {
                for (int k = 0; k < layerSizes[i + 1]; k++) {//for each in next layer
                    currentWSumGradients[i] += prevWSumGradients[k] * nodes[i + 1][k].getWeights()[&(nodes[i][j])];
                }//            ___  * ∂a_n/∂z_n (activationDerivative(z))
                currentWSumGradients[j] *= layerActivationDerivatives[i](nodes[i][j].getWeightedSum()); //multiply by ∂a_n/∂preAct_n
            } else {
                currentWSumGradients[j] = lossGradients[i] * layerActivationDerivatives[i](nodes[i][j].getWeightedSum()); //partial derivative of loss w/resp. preActivation val of node in output layer1`
            }
            nodes[i][j].updateBias(currentWSumGradients[i] * learningRate);

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

void NeuralNetwork::train(TrainingData data, int batchSize){
    if (batchSize <= 0 || batchSize > data.size()) {
        batchSize = data.size(); // Use all data if batchSize is invalid
    }
    std::vector<TrainingSample> samples(data.begin(), data.end()); //conver to vector

    std::random_device rd; //shuffle
    std::mt19937 gen(rd());
    std::shuffle(samples.begin(), samples.end(), gen);

    //iterate over batches
    for (std::size_t i = 0; i < samples.size(); i += batchSize){
        auto batchEnd = std::min(i + batchSize, samples.size()); //makes sure end of batch is within bounds of data
        std::vector<TrainingSample> batch(samples.begin() + i, samples.begin() + batchEnd); //create batch from i to batchEnd

        for (const auto& sample : batch) {//for each sample in batch
            backPropogate(sample);
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
        for (auto& layer : nodes){ 
            for (auto& node : layer){
                outFile << node.getBias() << " ";
                for (auto& pair : node.getWeights()){
                    outFile << pair.second << " ";
                }
                outFile << "\n";
            }
        }
        outFile.close();
    } else{
        std::cerr << "Unable to open file for writing" << std::endl;
    }
}

void NeuralNetwork::uploadParameters(std::string fileName, std::initializer_list<ActivationPair> layerActivations){
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
        for (int i = 0; i < layerCount; i++){
            inFile >> layerSizes[i];
        } //read all layerSizes

        Layer* prevLayer;
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