/* Written by Grant Allan - 5/30/2022
 * Last edited by Grant Allan  - 5/30/2022
 * 
 * Implementation of the logic for a neuron.
 * 
 * The basics of this were based off the tutorial at
 * https://www.youtube.com/watch?v=sK9AbJ4P8ao
 */

#include <cmath>
#include <vector>
#include <cstdlib>


struct Connection
{
    // Could have set of connected to x neurons with x weights for sparsely connected or recurrent nets
    // But we're just making a dense sequential model, so we don't need that
    double weight;
    double deltaWeight;
    double bias;
};


class Neuron;
typedef std::vector<Neuron> Layer;

/*
 * Neuron class
 */
class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer& prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer& nextLayer);
    void updateInputWeights(Layer& prevLayer);

private:
    static double learning_rate; // [0.0, 1.0], also called eta
    static double momentum; // [0.0, 1.0], also called alpha

    double m_outputVal;
    double m_gradient;
    unsigned m_index;
    std::vector<Connection> m_outputWeights;

    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }

    double sumDOW(const Layer &nextLayer) const;
};

double Neuron::learning_rate = 0.15;
double Neuron::momentum = 0.5;


Neuron::Neuron(unsigned numOutputs, unsigned index)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
        m_outputWeights.back().bias = randomWeight();
    }

    m_index = index;
}


void Neuron::feedForward(const Layer& prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (our inputs)
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += (prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_index].weight) +
                prevLayer[n].m_outputWeights[m_index].bias;
    } 
    m_outputVal = Neuron::activationFunction(sum);
}


void Neuron::updateInputWeights(Layer& prevLayer)
{
    // Update weights in the Connection container in the
    // neurons of the previous layer
    for (unsigned n = 0; n < prevLayer.size(); n++) {
        Neuron& neuron = prevLayer[n];

        double oldDeltaWeight = neuron.m_outputWeights[m_index].deltaWeight;
        double newDeltaWeight = // Individual input, magnified by the gradient and learning rate
                                // learning rate is also called eta
                                learning_rate * m_gradient * neuron.getOutputVal()
                                // Affected by momentum (a fraction of the previous delta weight)
                                // momentum is also called alpha
                                + momentum + oldDeltaWeight;

        neuron.m_outputWeights[m_index].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_index].weight += newDeltaWeight;
    }
}


double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    // Sum of our contributions of the errors at the nodes we feed
    for (unsigned n = 0; n < nextLayer.size() - 1; n++) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}


void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::activationFunctionDerivative(m_outputVal);
}


void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::activationFunctionDerivative(m_outputVal);
}


double Neuron::activationFunction(double x) {
    return tanh(x);
}


double Neuron::activationFunctionDerivative(double x) {
    return 1.0 - x * x;
}
