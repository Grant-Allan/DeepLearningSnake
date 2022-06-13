#include <Eigen/Eigen>
#include <iostream>
#include <vector>


/*
 * Support structs
 */
// Settings used in topology to build each layer
struct LayerSettings
{
    int units;
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
    Eigen::RowVectorXd simple_outputs;
    Eigen::RowVectorXd activated_outputs;
    std::string activation_function;
    //std::function<double(double)> activation_function;
    DenseLayer(unsigned prev_layer_size, unsigned cur_layer_size, std::string activation_function, unsigned &neuron_id);
private:
};


/*
 * The Optimizer used for the neural network's backpropagation.
 */
class Optimizer
{
public:
    double categorical_crossentropy(Eigen::RowVectorXd &y_pred, Eigen::RowVectorXd &y_true);
    Eigen::RowVectorXd categorical_crossentropy_derivative(Eigen::RowVectorXd &y_pred, Eigen::RowVectorXd &y_true);

    double mean_squared_error(Eigen::RowVectorXd &y_pred, Eigen::RowVectorXd &y_true);
    Eigen::RowVectorXd mean_squared_error_derivative(Eigen::RowVectorXd &y_pred, Eigen::RowVectorXd &y_true);
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
    Optimizer optimizer;
    Eigen::RowVectorXd input;

    // Model use
    Net(int input_size, std::vector<LayerSettings> topology, Optimizer opt);
    void predict(Eigen::RowVectorXd &input, Eigen::RowVectorXd &prediction);
    void train(Eigen::RowVectorXd &input, Eigen::RowVectorXd &y_true);
    void results(Eigen::RowVectorXd &prediction);
    void backpropagation(Eigen::RowVectorXd &y_pred, Eigen::RowVectorXd &y_true);

    // Activation functions
    Eigen::RowVectorXd Step(Eigen::RowVectorXd &x);

    Eigen::RowVectorXd ReLu(Eigen::RowVectorXd &x);
    Eigen::RowVectorXd ReLuDerivative(Eigen::RowVectorXd &x);

    Eigen::RowVectorXd Sigmoid(Eigen::RowVectorXd &x);
    Eigen::RowVectorXd SigmoidDerivative(Eigen::RowVectorXd &x);

    Eigen::RowVectorXd Tanh(Eigen::RowVectorXd &x);
    Eigen::RowVectorXd TanhDerivative(Eigen::RowVectorXd &x);

    Eigen::RowVectorXd Softmax(Eigen::RowVectorXd &x);
    Eigen::MatrixXd SoftmaxDerivative(Eigen::RowVectorXd &x);

private:
    void feedforward(Eigen::RowVectorXd &input, Eigen::RowVectorXd &output);
    void batch_update(std::vector<Eigen::MatrixXd> &d_weights, std::vector<Eigen::RowVectorXd> &d_biases);
};
