/* Written by Grant Allan - 5/30/2022
 * Last edited by Grant Allan  - 5/30/2022
 * 
 * Implementation of the logic for a Sequential neural network
 * with Dense layers.
 * 
 * The basics of this were based off the tutorial at
 * https://www.youtube.com/watch?v=sK9AbJ4P8ao
 */

#include "Neuron.cpp"
#include <iostream>
#include <cassert>
#include <cmath>


// Topology layer structure
struct LayerSet
{
    unsigned neurons;
    std::string layer_type;
    std::string activation_function;
};

typedef std::vector<LayerSet> topology;

/*
 * Sequential Neural Network with Dense layers
 */
class Net
{
public:
    Net(const std::vector<LayerSet>& topology);

    void feedForward(const std::vector<double>& inputVals);
    void backPropagation(const std::vector<double>& targetVals);
    void getResults(std::vector<double> &resultVals) const;

    double getRecentAverageError(void) const { return m_recentAverageError; }

private:
    static double m_recentAverageSmoothingFactor;
    double m_error;
    double m_recentAverageError;
    std::vector<Layer> m_layers;  // m_Layers[layerNum][neuronNum]
};

double Net::m_recentAverageSmoothingFactor = 100.0;


// NeuralNetwork constructor
Net::Net(const std::vector<LayerSet>& topology)
{
    // Length of topology is the number of layers
    for (unsigned layerNum = 0; layerNum < topology.size(); ++layerNum)
    {
        // Add current layer to the model
        m_layers.push_back(Layer());

        // Get the number of outputs, dependent on whether or not it's the output layer
        unsigned numOutputs = (layerNum == topology.size() - 1) ? 0 : topology[layerNum + 1].neurons;

        // Fill the Layer with neurons, with each neuron containing a weight and bias
        for (unsigned n = 0; n < topology[layerNum].neurons; n++) {
            m_layers.back().push_back(Neuron(numOutputs, n));
            std::cout << "Added neuron " << n << " to layer " << layerNum << '\n';
        }
        std::cout << '\n';

        // Set the bias to 1.0 (it's not updated and just stays at 1.0 as a constant)
        //m_layers.back().back().setOutputVal(1.0);
    }
}


void Net::feedForward(const std::vector<double>& inputVals)
{
    // Make sure we have the correct input size
    assert(inputVals.size() == m_layers[0].size());

    // Assign (latch) input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); i++) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // Forward propagation
    // For each layer
    for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
        Layer& prevLayer = m_layers[layerNum - 1];
        // For each neuron in the layer
        for (unsigned n = 0; n < m_layers[layerNum].size(); n++) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}


void Net::backPropagation(const std::vector<double>& targetVals)
{
    // Calculate overall net error (RMS of output neuron errors)
    // RMS = Root Mean Squared Error
    Layer& outputLayer = m_layers.back();
    m_error = 0.0;

    // Get the error of each output neuron
    for (unsigned n = 0; n < outputLayer.size(); n++) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    
    // Get the final loss value
    m_error /= outputLayer.size() - 1; // average error squared, bias not included in size
    m_error = sqrt(m_error); // RMS

    // Recent average measurement to see how well the model is being trained
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) /
                           (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients
    for (unsigned n = 0; n < outputLayer.size()-1; n++) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers
    for (unsigned layerNum = m_layers.size()-2; layerNum > 1; layerNum--)
    {
        Layer& curLayer = m_layers[layerNum];
        Layer& nextLayer = m_layers[layerNum + 1];

        // For each neuron in the current layer
        for (unsigned n = 0; n < curLayer.size(); n++) {
            curLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer, update connection weights
    for (unsigned layerNum = m_layers.size()-1; layerNum > 0; layerNum--)
    {
        Layer& curLayer = m_layers[layerNum];
        Layer& prevLayer = m_layers[layerNum - 1];

        // For each neuron in the current layer
        for (unsigned n = 0; n < curLayer.size()-1; n++) {
            curLayer[n].updateInputWeights(prevLayer);
        }
    }
}


// Get output values of the output layer
void Net::getResults(std::vector<double> &resultVals) const
{
    resultVals.clear();
    for (unsigned n = 0; n < m_layers.back().size(); ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}


void showVectorVals(std::string label, std::vector<double>& v)
{
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << '\n';
}