#include "NeuralNet.hpp"


// Neural Network constructor
Net::Net(int input_size, std::vector<LayerSettings> topology)
{
    // Save the topology and input size
    this->topology = topology;
    this->input_size = input_size;
    this->neuron_id = 0;
    Optimizer opt;
    this->optimizer = opt;

    // Build neural network using given topology
    // Create first layer
    this->layers.push_back(new DenseLayer(input_size, topology[0].units, topology[0].activation_function, this->neuron_id));

    // Create other layers
    for (int l = 1; l < topology.size(); l++)
    {
        // Select a layer type, then set the size and activation function
        if (topology[l].layer_type == "Dense" || topology[l].layer_type == "dense")
            this->layers.push_back(new DenseLayer(topology[l-1].units, topology[l].units, topology[l].activation_function, this->neuron_id));
    }
}


// Process an input to get an output
void Net::predict(std::vector<double> &input, std::vector<double> &output)
{
    Net::feedforward(input, output);
}

// Train the model on a given dataset
void Net::train(std::vector<double> &input, std::vector<double> &targets)
{
    std::vector<double> output(this->topology.back().units);
    Net::feedforward(input, output);
    Net::backpropagation(output, targets);
}


// Process an input to get an output
void Net::feedforward(std::vector<double> &input, std::vector<double> &output)
{
    Eigen::RowVectorXd inputs(this->input_size);
    for (int i = 0; i < input.size(); i++)
        inputs[i] = input[i];

    // Feed input into first hidden layer and get outputs
    if (input.size() == this->input_size) {
        for (int n = 0; n < this->topology[0].units; n++) {
            // References to make it easier to read
            Eigen::RowVectorXd &cur_value = this->layers[0]->neurons[n].values;
            Eigen::RowVectorXd &bias = this->layers[0]->neurons.back().values;

            // Apply activation function
            if (this->layers[0]->activation_function == "ReLu" || this->layers[0]->activation_function == "relu") {
                this->layers[0]->outputs[n] = Net::ReLu(inputs, cur_value, bias[n]);
            } else if (this->layers[0]->activation_function == "Step" || this->layers[0]->activation_function == "step") {
                this->layers[0]->outputs[n] = Net::Step(inputs, cur_value, bias[n]);
            } else if (this->layers[0]->activation_function == "Sigmoid" || this->layers[0]->activation_function == "sigmoid") {
                this->layers[0]->outputs[n] = Net::Sigmoid(inputs, cur_value, bias[n]);
            } else if (this->layers[0]->activation_function == "Tanh" || this->layers[0]->activation_function == "tanh") {
                this->layers[0]->outputs[n] = Net::Tanh(inputs, cur_value, bias[n]);
            } else if (this->layers[0]->activation_function == "Softmax" || this->layers[0]->activation_function == "softmax") {
                this->layers[0]->outputs[n] = cur_value.dot(inputs) + bias[n];
            }
        }
        // Set layer output to our newly obtained values
        if (this->layers[0]->activation_function == "Softmax" || this->layers[0]->activation_function == "softmax") {
            this->layers[0]->outputs = Net::Softmax(this->layers[0]->outputs);
        }
    } else {
        std::cout << "Wrong input size (expected size: "
                  << this->input_size << ", received size: "
                  << input.size() << ")" << '\n';
        exit(0);
    }

    // Feed input from each hidden layer into the next hidden layer
    for (int l = 1; l < this->topology.size(); l++) {
        // Apply dot product to each neuron and the previous layer's output
        for (int n = 0; n < this->topology[l].units; n++) {
            // References to make it easier to read
            Eigen::RowVectorXd &prev_output = this->layers[l-1]->outputs;
            Eigen::RowVectorXd &cur_value = this->layers[l]->neurons[n].values;
            Eigen::RowVectorXd &bias = this->layers[l]->neurons.back().values;

            // Apply activation function
            if (this->layers[l]->activation_function == "ReLu" || this->layers[l]->activation_function == "relu") {
                this->layers[l]->outputs[n] = Net::ReLu(cur_value, prev_output, bias[n]);
            } else if (this->layers[l]->activation_function == "Step" || this->layers[l]->activation_function == "step") {
                this->layers[l]->outputs[n] = Net::Step(cur_value, prev_output, bias[n]);
            } else if (this->layers[l]->activation_function == "Sigmoid" || this->layers[l]->activation_function == "sigmoid") {
                this->layers[l]->outputs[n] = Net::Sigmoid(cur_value, prev_output, bias[n]);
            } else if (this->layers[l]->activation_function == "Tanh" || this->layers[l]->activation_function == "tanh") {
                this->layers[l]->outputs[n] = Net::Tanh(cur_value, prev_output, bias[n]);
            } else if (this->layers[l]->activation_function == "Softmax" || this->layers[l]->activation_function == "softmax") {
                this->layers[l]->outputs[n] = cur_value.dot(prev_output) + bias[n];
            }
        }
        // Set layer output to our newly obtained values
        if (this->layers[l]->activation_function == "Softmax" || this->layers[l]->activation_function == "softmax") {
            this->layers[l]->outputs = Net::Softmax(this->layers[l]->outputs);
        }
    }

    // Fill output with the value of the output neurons
    std::vector<double> _output;
    for (int n = 0; n < this->topology.back().units; n++) {
        // Get the value of the neuron and add it to the output values
        _output.push_back(this->layers.back()->outputs[n]);
    }
    output = _output;
}


// Use an output and loss function to update the weights
void Net::backpropagation(std::vector<double> &output, std::vector<double> &targets)
{
    double learning_rate = 0.001;

    // Calculate layer error

    // Update layer weights
}


// Get the current output values of the neurons in the output layer
// Meant to be used after running Net::train()
void Net::results(std::vector<double> &output)
{
    // Fill output with the value of the output neurons
    for (int n = 0; n < this->topology.back().units; n++) {
        // Get the value of the neuron and add it to the output values
        output[n] = this->layers.back()->outputs[n];
    }
}



/*
 * Activation functions
 */
//
// ReLu
double Net::ReLu(Eigen::RowVectorXd cur_value, Eigen::RowVectorXd prev_output, double bias)
{
    // 0 or x, where x is cur_value.dot(prev_output) + bias)
    return ((cur_value.dot(prev_output) + bias) > 0) ? (cur_value.dot(prev_output) + bias) : 0;
}

//
// Step
double Net::Step(Eigen::RowVectorXd cur_value, Eigen::RowVectorXd prev_output, double bias)
{
    // 0 or 1
    return ((cur_value.dot(prev_output) + bias) > 0) ? 1 : 0;
}

//
// Sigmoid
double Net::Sigmoid(Eigen::RowVectorXd cur_value, Eigen::RowVectorXd prev_output, double bias)
{
    // 0 to 1
    return 1 / (1 + std::exp(cur_value.dot(prev_output) + bias));
}

double SigmoidDerivative(Eigen::RowVectorXd cur_value, Eigen::RowVectorXd prev_output, double bias)
{
    double x = 1 / (1 + std::exp(cur_value.dot(prev_output) + bias));
    return x * (1 - x);
}

//
// Tanh
double Net::Tanh(Eigen::RowVectorXd cur_value, Eigen::RowVectorXd prev_output, double bias)
{
    // -1 to 1
    return tanh(cur_value.dot(prev_output) + bias);
}

double Net::TanhDerivative(Eigen::RowVectorXd cur_value, Eigen::RowVectorXd prev_output, double bias)
{
    double x = tanh(cur_value.dot(prev_output) + bias);
    return 1 - (x*x);
}

//
// Softmax
Eigen::RowVectorXd Net::Softmax(Eigen::RowVectorXd x)
{
    // 0 to 1, where the values sum to 1
    Eigen::RowVectorXd e_x = x.array().log();
    return e_x.array() / e_x.sum();
}

Eigen::MatrixXd Net::SoftmaxDerivative(Eigen::RowVectorXd x)
{
    // Create array of zeros with a diagonal of ones
    int size = this->topology.back().units;
    Eigen::MatrixXd ones = Eigen::MatrixXd::Constant(size, 1, 1.0);
    Eigen::MatrixXd I = Eigen::MatrixXd::Constant(size, size, 0.0);
    I = ones.asDiagonal();

    // Store softmax so we don't have to calculate it twice
    Eigen::RowVectorXd softmax =  this->Softmax(x);

    // Return gradient
    return (I.rowwise() - softmax).array().colwise() * softmax.transpose().array();
}



/*
 * Layer constructors
 */
DenseLayer::DenseLayer(unsigned prev_layer_size, unsigned size, std::string activation_function, unsigned &neuron_id)
{
    // Create weight neurons
    std::vector<Neuron> neurons(size);
    for (Neuron &n : neurons) {
        Eigen::RowVectorXd values(prev_layer_size);
        for (unsigned w = 0; w < prev_layer_size; w++) {
            values[w] = rand() / double(RAND_MAX);
        }

        n.values = values;
        n.id = neuron_id++;
    }

    // Create bias neuron and add it to the weights
    Neuron bias;
    Eigen::RowVectorXd values(size);
    for (unsigned b = 0; b < size; b++) {
        values[b] = rand() / double(RAND_MAX);
    }
    bias.values = values;
    bias.id = neuron_id++;
    neurons.push_back(bias);

    // Create output layer
    Eigen::RowVectorXd outputs(size);
    this->outputs = outputs;

    // Set neurons and activation function
    this->neurons = neurons;
    this->activation_function = activation_function;
}



/*
 * Optimizer
 */
double Optimizer::categorical_crossentropy(Eigen::RowVectorXd y_pred, Eigen::RowVectorXd y_true)
{
    return -(y_true.dot(y_pred.array().log().matrix()));
}

double Optimizer::mean_squared_error(Eigen::RowVectorXd y_pred, Eigen::RowVectorXd y_true)
{
    return (((y_true - y_pred).array().square()) / y_pred.rows()).sum();
}


