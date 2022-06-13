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

    Eigen::RowVectorXd input;
    Eigen::RowVectorXd output;
    Eigen::RowVectorXd y_true;

    input.resize(input_size);
    y_true.resize(1);

    //net.predict(input, output);
    //std::cout << "Output: " << output[0] << '\n' << '\n';

    //net.backpropagation(output, y_true);

    // Train the model to be an XOR gate
    for (int i = 0; i < 10; i++)
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
    }
}