#include <Eigen/Eigen>
#include <iostream>
#include <vector>


/*
 * Support structs
 */
// Settings used in topology to build each layer
struct LayerSettings
{
    unsigned units;
    std::string layer_type;
    std::string activation_function;
};

// The neuron used in the layers
struct Neuron
{
    unsigned id;

    // Need to make this more complex so each connection
    // has a weight. So it would be a (connection, weight)
    // pair value. Possibly add bias for (conn, weight, bias)
    Eigen::RowVectorXd values;
    std::vector<unsigned> connections;
};


/*
 * Layers
 */
// Parent layer used to create a vector of layers
class Layer{};

// Layer of densely connected neurons
class DenseLayer : public Layer
{
public:
    std::vector<Neuron> neurons;
    Eigen::RowVectorXd outputs;
    std::string activation_function;
    //std::function<double(double)> activation_function;
    DenseLayer(unsigned prev_layer_size, unsigned size, std::string activation_function, unsigned &neuron_id);
private:
};


/*
 * The neural network itself
 */
class Net
{
public:
    std::vector<LayerSettings> topology;
    std::vector<DenseLayer*> layers;
    std::string loss_function;
    unsigned input_size;
    unsigned neuron_id;

    // Model use
    Net(unsigned input_size, std::vector<LayerSettings> topology);
    void predict(std::vector<double> &input, std::vector<double> &output);
    void train(std::vector<double> &input, std::vector<double> &targets);
    void results(std::vector<double> &output);

    // Activation functions
    double ReLu(Eigen::RowVectorXd cur_value, Eigen::RowVectorXd prev_output, double bias);
    double Step(Eigen::RowVectorXd cur_value, Eigen::RowVectorXd prev_output, double bias);
    double Sigmoid(Eigen::RowVectorXd cur_value, Eigen::RowVectorXd prev_output, double bias);
    double Tanh(Eigen::RowVectorXd cur_value, Eigen::RowVectorXd prev_output, double bias);
    Eigen::RowVectorXd Softmax(Eigen::RowVectorXd x);

private:
    void feedforward(std::vector<double> &input, std::vector<double> &output);
    void backpropagation(std::vector<double> &output, std::vector<double> &targets);
};
