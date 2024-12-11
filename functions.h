#include <vector>

const double lReluAlpha = 0.1;

//activation functions

using std::tanh;  //tanh activation

double sigmoid(double input);

double relu(double input);

double leakyRelu(double input);

//activation derivatives

double sigmoidDerivative(double input);

double tanhDerivative(double input);

double reluDerivative(double input);

double leakyReluDerivative(double input);


//loss functions

double lossMSE(std::vector<double> actual, std::vector<double> expected); //loss for regression

double lossBinaryCrossEntropy(std::vector<double> actual, std::initializerlist<double> expected); //loss for binary categorization (assuming sigmoid)

double lossMSEderivative(std::vector<double> actual, std::vector<double> expected, size_t wRespectIndex);

double lossBinaryCrossEntropyDerivative(std::vector<double> actual, std::vector<double> expected, size_t wRespectIndex);