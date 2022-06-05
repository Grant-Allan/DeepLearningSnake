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
    Eigen::RowVectorXd values;
    unsigned id;
    std::vector<unsigned> connections;

    // Set a (-1, 1) random value
    void randomValues(void) { this->values.setRandom(); }
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
    DenseLayer(unsigned prev_layer_size, unsigned size, std::string activation_function);
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
    static unsigned neuron_id;

    Net(unsigned input_size, std::vector<LayerSettings> topology);
    void predict(std::vector<double> &input, std::vector<double> &output);
    void train(std::vector<double> &input, std::vector<double> &targets);
    void results(std::vector<double> &output);

private:
    void feedforward(std::vector<double> &input, std::vector<double> &output);
    void backpropagation(std::vector<double> &output, std::vector<double> &targets);
};
