#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include "functions.h"
#include "neuralNetwork.h"
#include "mnist.h"

void runXor(){
    TrainingData xorData = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}},
    };

    NeuralNetwork network({2, 4, 1},
        {{sigmoid, sigmoidDerivative}, {tanh, tanhDerivative}, {sigmoid, sigmoidDerivative}},
        lossMSE, lossMSEderivative, 0.5);

    for (int epoch = 0; epoch < 5000; epoch++){
        network.train(xorData, 1);
    }

    for (auto& dataPoint : xorData){
        network.evaluate(dataPoint.first);
        std::cout << dataPoint.first[0] << " XOR " << dataPoint.first[1]
                  << " -> " << network.output()[0] << " (expected " << dataPoint.second[0] << ")\n";
    }
}

//helper for % samples where highest output matches labeled digit
double accuracy(NeuralNetwork& network, const TrainingData& dataSet){
    int correct = 0;
    for (auto& dataPoint : dataSet){
        network.evaluate(dataPoint.first);
        if (indexMax(network.output()) == indexMax(dataPoint.second)){
            correct++;
        }
    }
    return correct * 100.0 / dataSet.size();
}

void showPredictions(NeuralNetwork& network, const TrainingData& testData, int count){
    int shown = 0;
    for (auto& dataPoint : testData){
        if (shown++ == count) break;
        network.evaluate(dataPoint.first);
        std::cout << "test image " << shown << ": predicted " << indexMax(network.output())
                  << ", actual " << indexMax(dataPoint.second) << "\n";
    }
}

const int trainSamples = 60000;
const int testSamples = 10000;
const int epochs = 5;

TrainingData loadMnistTrain(const std::string& dataDir){
    return loadMnistData(dataDir + "/train-images-idx3-ubyte", dataDir + "/train-labels-idx1-ubyte", trainSamples);
}

TrainingData loadMnistTest(const std::string& dataDir){
    return loadMnistData(dataDir + "/t10k-images-idx3-ubyte", dataDir + "/t10k-labels-idx1-ubyte", testSamples);
}

void trainMnist(NeuralNetwork& network, const TrainingData& trainData, const TrainingData& testData){
    std::cout << trainData.size() << " training samples, " << testData.size() << " test samples\n";
    std::cout << "test accuracy before training: " << accuracy(network, testData) << "%\n";

    for (int epoch = 1; epoch <= epochs; epoch++){
        network.train(trainData, 1); //one pass over shuffled training set
        std::cout << "epoch " << epoch << ": test accuracy " << accuracy(network, testData) << "%\n";
    }

    showPredictions(network, testData, 10);
}

//each layer has its own activation: tanh, then leaky relu, then sigmoid at the output
//the first pair is only a placeholder for the input layer
//loading saved parameters needs the same activations, so runMixedMnist and runSavedMnist share this list
const std::initializer_list<ActivationPair> mixedActivations = {
    {sigmoid, sigmoidDerivative}, {tanh, tanhDerivative}, {leakyRelu, leakyReluDerivative}, {sigmoid, sigmoidDerivative}
};

//trains the network, then saves the trained parameters to parameters.txt
void runMixedMnist(const std::string& dataDir){
    std::cout << "loading MNIST training and testing data from " << dataDir << "...\n";
    TrainingData trainData = loadMnistTrain(dataDir);
    TrainingData testData = loadMnistTest(dataDir);

    NeuralNetwork network({784, 32, 16, 10}, mixedActivations, lossMSE, lossMSEderivative, 0.05); //leaky relu needs smaller steps than sigmoid
    trainMnist(network, trainData, testData);

    network.downloadParameters();
    std::cout << "saved trained parameters to parameters.txt\n";
}

//mnist network without training: loads the parameters saved by runMixedMnist
void runSavedMnist(const std::string& dataDir){
    std::cout << "loading MNIST testing data from " << dataDir << "...\n";
    TrainingData testData = loadMnistTest(dataDir);

    NeuralNetwork network({784, 32, 16, 10}, mixedActivations, lossMSE, lossMSEderivative);
    network.uploadParameters("parameters.txt", mixedActivations);

    //should match the last epoch of runMixedMnist (parameters are saved with 6 digits, so maybe off by a little)
    std::cout << "test accuracy with saved parameters: " << accuracy(network, testData) << "%\n";
    showPredictions(network, testData, 10);
}

//prints how many seconds have passed since start, to 0.1 seconds
void printElapsed(const std::string& name, std::chrono::steady_clock::time_point start){
    std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;
    std::cout << name << " took " << std::fixed << std::setprecision(1) << seconds.count() << " seconds\n"
              << std::defaultfloat << std::setprecision(6); //restore default formatting
}

int main(int argc, char* argv[]){
    std::string dataDir = argc > 1 ? argv[1] : "data/mnist"; //optionally pass MNIST folder as the first arg

    // wait here so I can set up Windows perfmon to capture this process
    // std::cout << "==== Type any key to continue ====\n";
    // char c;
    // std::cin >> c;

    std::cout << "=== XOR ===\n";
    auto start = std::chrono::steady_clock::now();
    runXor();
    printElapsed("runXor", start);

    std::cout << "\n=== MNIST, mixed activations ===\n";
    start = std::chrono::steady_clock::now();
    runMixedMnist(dataDir);
    printElapsed("runMixedMnist", start);

    std::cout << "\n=== MNIST with saved parameters ===\n";
    start = std::chrono::steady_clock::now();
    runSavedMnist(dataDir);
    printElapsed("runSavedMnist", start);
    return 0;
}
