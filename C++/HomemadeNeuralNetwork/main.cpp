#include "NeuralNet.cpp"


int main()
{
    int input_size = 2;
    std::vector<LayerSettings> topology;

    // Build topology
    LayerSettings layer;
    
    layer.units = 6;
    layer.layer_type = "Dense";
    layer.activation_function = "ReLu";
    topology.push_back(layer);
    
    layer.units = 4;
    layer.layer_type = "Dense";
    layer.activation_function = "ReLu";
    topology.push_back(layer);
    
    layer.units = 2;
    layer.layer_type = "Dense";
    layer.activation_function = "ReLu";
    topology.push_back(layer);
    
    layer.units = 1;
    layer.layer_type = "Dense";
    layer.activation_function = "Sigmoid";
    topology.push_back(layer);

    Optimizer opt;
    Net net(input_size, topology, opt);
    net.learning_rate = 0.001;

    Eigen::RowVectorXd input;
    Eigen::RowVectorXd output;
    Eigen::RowVectorXd y_true;

    input.resize(input_size);
    y_true.resize(1);

    // Set max_accum_loss to make sure the model ends up well trained
    net.max_accum_loss = 16;

    // Add value to accum_loss so we get through at least one set of accumaluated losses
    net.accumulated_loss(1000000000);

    // Train the model to be an XOR gate
    int i = 0;
    while (net.avg_loss > 0.000000001)
    {
        std::cout << '\n' << "=============================" << '\n';
        std::cout << '\n' << "Episode: " << i << '\n';

        if (i % 4 == 0) {
            input[0] = 0;
            input[1] = 0;
            y_true[0] = 0;
        } else if ((i+1) % 4 == 0) {
            input[0] = 0;
            input[1] = 1;
            y_true[0] = 1;
        } else if ((i+2) % 4 == 0) {
            input[0] = 1;
            input[1] = 0;
            y_true[0] = 1;
        } else if ((i+3) % 4 == 0) {
            input[0] = 1;
            input[1] = 1;
            y_true[0] = 0;
        }

        net.train(input, y_true);
        net.results(output);

        std::cout << "Input: " << input[0] << " " << input[1] << '\n';
        std::cout << "Prediction: " << output[0] << '\n';
        std::cout << "Target: " << y_true[0] << '\n';
        i++;
    }
    std::cout << '\n' << "=============================" << '\n';
}