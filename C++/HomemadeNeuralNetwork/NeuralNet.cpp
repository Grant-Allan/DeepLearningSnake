#include "NeuralNet.hpp"
#include <numeric>


// Neural Network constructor
Net::Net(int input_size, std::vector<LayerSettings> topology, Optimizer opt)
{
    // Save the topology and input size
    this->topology = topology;
    this->input_size = input_size;
    this->neuron_id = 0;
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
void Net::predict(Eigen::RowVectorXd &input, Eigen::RowVectorXd &output)
{
    Net::feedforward(input, output);
}

// Train the model on a given dataset
void Net::train(Eigen::RowVectorXd &input, Eigen::RowVectorXd &y_true)
{
    Eigen::RowVectorXd output(topology.back().units);
    Net::feedforward(input, output);
    Net::backpropagation(output, y_true);
}

// Apply the designated activation function to the given input
Eigen::RowVectorXd Net::applyActivationFunction(std::string actFuncName, Eigen::RowVectorXd &simple_outputs)
{
    if (actFuncName == "ReLu" || actFuncName == "relu") return Net::ReLu(simple_outputs);
    else if (actFuncName == "Sigmoid" || actFuncName == "sigmoid") return Net::Sigmoid(simple_outputs);
    else if (actFuncName == "Tanh" || actFuncName == "tanh") return Net::Tanh(simple_outputs);
    else if (actFuncName == "Softmax" || actFuncName == "softmax") return Net::Softmax(simple_outputs);
}

Eigen::RowVectorXd Net::applyActivationFunctionDerivative(std::string actFuncName, Eigen::RowVectorXd &simple_outputs)
{
    if (actFuncName == "ReLu" || actFuncName == "relu") return Net::ReLuDerivative(simple_outputs);
    else if (actFuncName == "Sigmoid" || actFuncName == "sigmoid") return Net::SigmoidDerivative(simple_outputs);
    else if (actFuncName == "Tanh" || actFuncName == "tanh") return Net::TanhDerivative(simple_outputs);
}

// Process an input to get an output
void Net::feedforward(Eigen::RowVectorXd &inputs, Eigen::RowVectorXd &prediction)
{
    // Set internal input
    this->input = inputs;

    // Feed input into first hidden layer and get outputs
    if (inputs.array().cols() == this->input_size) {
        // References to make it easier to read
        Eigen::RowVectorXd &bias = this->layers[0]->neurons.back().values;
        Eigen::RowVectorXd &simple_outputs = this->layers[0]->simple_outputs;
        Eigen::RowVectorXd &activated_outputs = this->layers[0]->activated_outputs;
        for (int n = 0; n < this->topology[0].units; n++) {
            // References to make it easier to read
            Eigen::RowVectorXd &cur_value = this->layers[0]->neurons[n].values;

            // Apply activation function
            simple_outputs[n] = cur_value.dot(inputs) + bias[n];
        }
        // Set layer output to our newly obtained values
        activated_outputs = applyActivationFunction(this->layers[0]->activation_function, simple_outputs);
    } else {
        std::cout << "Wrong input size (expected size: "
                  << this->input_size << ", received size: "
                  << inputs.cols() << ")" << '\n';
        exit(0);
    }

    // Feed input from each hidden layer into the next hidden layer
    for (int l = 1; l < this->topology.size(); l++) {
        // References to make it easier to read
        Eigen::RowVectorXd &prev_output = this->layers[l-1]->activated_outputs;
        Eigen::RowVectorXd &cur_simple_outputs = this->layers[l]->simple_outputs;
        Eigen::RowVectorXd &cur_activated_outputs = this->layers[l]->activated_outputs;
        Eigen::RowVectorXd &bias = this->layers[l]->neurons.back().values;

        // Apply dot product to each neuron and the previous layer's output
        for (int n = 0; n < this->topology[l].units; n++) {
            // References to make it easier to read
            Eigen::RowVectorXd &cur_value = this->layers[l]->neurons[n].values;

            // Apply Get output function
            cur_simple_outputs[n] = cur_value.dot(prev_output) + bias[n];
        }
        // Set layer output to our newly obtained values
        cur_activated_outputs = applyActivationFunction(this->layers[l]->activation_function, cur_simple_outputs);
    }

    // Set final output
    prediction = this->layers.back()->activated_outputs;
}


// Calculate accumulated loss
void Net::accumulated_loss(double loss_)
{
    // Add current loss to the accumultion
    if (this->accum_loss.size() < this->max_accum_loss) {
        this->accum_loss.push_back(loss_);
        this->cur_loss = this->accum_loss.size()-1;
    } else {
        this->accum_loss[this->cur_loss] = loss_;
        this->cur_loss++;

        // Reset cur_loss, if needed
        if (this->cur_loss >= this->max_accum_loss) {
            this->cur_loss = 0;
        }
    }

    // Recalculate avg_loss
    this->avg_loss = std::accumulate(this->accum_loss.begin(), this->accum_loss.end(), 0.0) / this->accum_loss.size();
}


// Use an output and loss function to update the weights
void Net::backpropagation(Eigen::RowVectorXd &y_pred, Eigen::RowVectorXd &y_true)
{
    // Current loss
    this->loss = this->optimizer.mean_squared_error(y_pred, y_true);
    std::cout << "y_pred: " << y_pred << '\n';
    std::cout << "y_true: " << y_true << '\n';
    std::cout << "Loss: " << this->loss << '\n' << '\n';

    // Calculate accumulated loss
    this->accumulated_loss(this->loss);
    //std::cout << "Average Loss: " << this->avg_loss << '\n' << '\n';

    // The changes we'll make to the weights and biases
    std::vector<Eigen::MatrixXd> d_weights;
    std::vector<Eigen::RowVectorXd> d_biases;

    // Initialize d_weights and d_biases
    for (int l = 0; l < this->topology.size(); l++)
    {
        int rows = (l > 0) ? this->layers[l-1]->neurons.size()-1 : this->input_size;
        Eigen::MatrixXd weight = Eigen::MatrixXd::Constant(rows, this->layers[l]->neurons.size()-1, 0.0);
        Eigen::RowVectorXd bias = Eigen::RowVectorXd::Constant(this->layers[l]->neurons.back().values.cols(), 0.0);

        //std::cout << rows << " " << this->layers[l]->neurons.size()-1 << '\n' << '\n';

        d_weights.push_back(weight);
        d_biases.push_back(bias);
        //std::cout << "d_weights " << l << ":" << '\n' << d_weights.back() << '\n' << '\n';
        //std::cout << "d_biases " << l << ":" << '\n' << d_biases.back() << '\n' << '\n';
    }
    //std::cout << "===========================================" << '\n' << '\n';

    //
    // Output layer
    //

    // References
    Eigen::RowVectorXd &simple_outputs = this->layers.back()->simple_outputs;
    Eigen::RowVectorXd &activated_outputs = this->layers[this->topology.size()-2]->activated_outputs;

    // Calculations
    Eigen::RowVectorXd d_error = this->optimizer.mean_squared_error_derivative(y_pred, y_true);
    Eigen::RowVectorXd delta = d_error * applyActivationFunctionDerivative(this->layers.back()->activation_function, simple_outputs);
    d_weights.back() += (activated_outputs.transpose() * delta);
    d_biases.back() += delta;

    //std::cout << "d_weights " << this->layers.size()-1 << ":" << '\n' << d_weights.back() << '\n' << '\n';
    //std::cout << "d_bias " << this->layers.size()-1 << ":" << '\n' << d_biases.back() << '\n' << '\n';
    //std::cout << "===========================================" << '\n' << '\n';

    //
    // Hidden layers
    //
    for (int l = this->layers.size()-2; l >= 0; l--) {
        // References
        Eigen::RowVectorXd &simple_outputs = this->layers[l]->simple_outputs;
        Eigen::RowVectorXd &activated_outputs = (l > 0) ? this->layers[l-1]->activated_outputs : this->input;

        // Calculations
        Eigen::RowVectorXd delta = d_error * applyActivationFunctionDerivative(this->layers[l]->activation_function, simple_outputs);

        d_weights[l] += learning_rate * (activated_outputs.transpose() * delta);
        d_biases[l] += learning_rate * delta;

        //std::cout << "d_weights " << l << ":" << '\n' << d_weights[l] << '\n' << '\n';
        //std::cout << "d_bias " << l << ":" << '\n' << d_biases[l] << '\n' << '\n';
        //std::cout << "===========================================" << '\n' << '\n';
    }

    this->batch_update(d_weights, d_biases);
}


void Net::batch_update(std::vector<Eigen::MatrixXd> &d_weights, std::vector<Eigen::RowVectorXd> &d_biases)
{
    // Update the neurons of each layer
    for (int l = 0; l < topology.size(); l++) {
        // Update weight neurons
        //std::cout << "weights " << l << ":" << '\n';
        for (int n = 0; n < topology[l].units; n++) {
            Eigen::RowVectorXd &neuron = this->layers[l]->neurons[n].values;
            const Eigen::RowVectorXd &d_neuron = d_weights[l].col(n);
            neuron -= d_neuron;

            //std::cout << neuron << '\n'; // weight
        }

        // Update bias neuron
        Eigen::RowVectorXd &neuron = this->layers[l]->neurons.back().values;
        neuron -= d_biases[l];

        //std::cout << '\n' << "bias " << l << ":" << '\n' << neuron << '\n' << '\n'; // bias
    }
}


// Get the current output values of the neurons in the output layer
// Meant to be used after running Net::train()
void Net::results(Eigen::RowVectorXd &prediction)
{
    // Set the output
    prediction = this->layers.back()->activated_outputs;
}



/*
 * Activation functions
 */
//
// ReLu
Eigen::RowVectorXd Net::ReLu(Eigen::RowVectorXd &x)
{
    // 0 or x, where x is cur_value.dot(prev_output) + bias)
    for(auto &w : x) w = (w > 0) ? w : 0;
    return x;
}

Eigen::RowVectorXd Net::ReLuDerivative(Eigen::RowVectorXd &x)
{
    // 0 or x, where x is cur_value.dot(prev_output) + bias)
    for(auto &w : x) w = (w > 0) ? 1 : 0;
    return x;
}

//
// Sigmoid
Eigen::RowVectorXd Net::Sigmoid(Eigen::RowVectorXd &x)
{
    // 0 to 1
    for(auto &w : x) w = 1 / (1 + std::exp(-w));
    return x;
}

Eigen::RowVectorXd Net::SigmoidDerivative(Eigen::RowVectorXd &x)
{
    for(auto &w : x) {
        w = 1 / (1 + std::exp(w));
        w = w * (1 - w);
    }
    return x;
}

//
// Tanh
Eigen::RowVectorXd Net::Tanh(Eigen::RowVectorXd &x)
{
    // -1 to 1
    for(auto &w : x) w = tanh(w);
    return x;
}

Eigen::RowVectorXd Net::TanhDerivative(Eigen::RowVectorXd &x)
{
    for(auto &w : x) {
        w = tanh(w);
        w = 1 - (w * w);
    }
    return x;
}

//
// Softmax
Eigen::RowVectorXd Net::Softmax(Eigen::RowVectorXd &x)
{
    // 0 to 1, where the values sum to 1
    Eigen::RowVectorXd e_x = x.array().log();
    return e_x.array() / e_x.sum();
}

Eigen::MatrixXd Net::SoftmaxDerivative(Eigen::RowVectorXd &x)
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
DenseLayer::DenseLayer(unsigned prev_layer_size, unsigned cur_layer_size, std::string activation_function, unsigned &neuron_id)
{
    // Create weight neurons
    std::vector<Neuron> neurons(cur_layer_size);
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
    Eigen::RowVectorXd values(cur_layer_size);
    for (unsigned b = 0; b < cur_layer_size; b++) {
        values[b] = rand() / double(RAND_MAX);
    }
    bias.values = values;
    bias.id = neuron_id++;
    neurons.push_back(bias);

    // Create output layer
    Eigen::RowVectorXd outputs(cur_layer_size);
    this->simple_outputs = outputs;
    this->activated_outputs = outputs;

    // Set neurons and activation function
    this->neurons = neurons;
    this->activation_function = activation_function;
}



/*
 * Optimizers
 * NOTE: if y_hat is ever used, it's the same as y_pred
 */
//
// Categorical Cross Entropy
double Optimizer::categorical_crossentropy(Eigen::RowVectorXd &y_pred, Eigen::RowVectorXd &y_true)
{
    return -(y_true.dot(y_pred.array().log().matrix()));
}

Eigen::RowVectorXd Optimizer::categorical_crossentropy_derivative(Eigen::RowVectorXd &y_pred, Eigen::RowVectorXd &y_true)
{
    return -(y_pred.array() / y_true.array());
}

//
// Squared Error
double Optimizer::squared_error(Eigen::RowVectorXd &y_pred, Eigen::RowVectorXd &y_true)
{
    return ((y_true - y_pred).array().square()).sum() / 2;
}

Eigen::RowVectorXd Optimizer::squared_error_derivative(Eigen::RowVectorXd &y_pred, Eigen::RowVectorXd &y_true)
{
    return y_pred.array() - y_true.array();
}

//
// Mean Squared Error
double Optimizer::mean_squared_error(Eigen::RowVectorXd &y_pred, Eigen::RowVectorXd &y_true)
{
    return ((y_true - y_pred).array().square()).sum() / y_true.cols();
}

Eigen::RowVectorXd Optimizer::mean_squared_error_derivative(Eigen::RowVectorXd &y_pred, Eigen::RowVectorXd &y_true)
{
    return 2 * (y_pred.array() - y_true.array()) / y_true.cols();
}


