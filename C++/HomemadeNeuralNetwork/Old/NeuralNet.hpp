#include <Eigen/Eigen>
#include <iostream>
#include <vector>


typedef Eigen::MatrixXd Matrix;
typedef Eigen::RowVectorXd RowVector;
typedef Eigen::VectorXd ColVector;
 
// Neural Network implementation
class NeuralNetwork {
public:
    // Constructor
    NeuralNetwork(std::vector<unsigned> topology, float learning_rate=0.001f);
 
    // Get an output through forward propagation
    void forwardPropagation(RowVector& input);
 
    // Calculate each layer's error
    void calcErrors(RowVector& output);
 
    // Update weights and biases using the previously calculated error/loss
    void updateWeights();
 
    // Backpropagate through the network, updating weights and biases
    void backpropagation(RowVector& output);
 
    // Training functions
    void train_batch(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data);
    void train(RowVector& input_data, RowVector& output_data);
 
    /* Storage objects for working of neural network
     * 
     * std::vector<Class> calls the destructor of whatever Class you're using after it's
     * pushed back. We use pointers so that it can't do that. Also, because we can use less
     * memory that way.
     */
    std::vector<unsigned> topology; // stores the number of neurons in each layer
    std::vector<RowVector*> neuronLayers; // stores the different layers of out network
    std::vector<RowVector*> cacheLayers; // stores the unactivated (activation fn not yet applied) values of layers
    std::vector<RowVector*> deltas; // stores the error contribution of each neurons
    std::vector<Matrix*> weights; // the connection weights itself
    float learning_rate;
};




