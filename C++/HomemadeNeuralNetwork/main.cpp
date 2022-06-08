#include "NeuralNet.cpp"


int main()
{
    unsigned input_size = 2;
    std::vector<LayerSettings> topology;

    // Build topology
    LayerSettings layer;
    
    layer.units = 4;
    layer.layer_type = "Dense";
    layer.activation_function = "ReLu";
    topology.push_back(layer);
    
    layer.units = 2;
    layer.layer_type = "Dense";
    layer.activation_function = "Softmax";
    topology.push_back(layer);

    Net net(input_size, topology);

    std::vector<double> input;
    std::vector<double> targets;
    std::vector<double> output;

    input.push_back(0);
    input.push_back(0);
    targets.push_back(0);

    net.predict(input, output);
    std::cout << "Output: " << output[0] << " " << output[1] << '\n' << '\n';

    Optimizer opt;

    Eigen::RowVectorXd _outputs;
    Eigen::RowVectorXd _targets;

    _outputs.resize(2);
    _targets.resize(2);

    _outputs[0] = output[0];
    _outputs[1] = output[1];

    _targets[0] = 0;
    _targets[1] = 1;

    opt.categorical_crossentropy(_outputs, _targets);
    std::cout << "Categorical Cross Entropy: " << opt.categorical_crossentropy(_outputs, _targets) << '\n' << '\n';

    /*
    // Train the model to be an XOR gate
    for (int i = 0; i < 10000; i++)
    {
        std::cout << '\n' << "Episode: " << i << '\n';

        if (i % 4 == 0)
        {
            input.push_back(0);
            input.push_back(0);
            targets.push_back(0);
        }
        else if ((i+1) % 4 == 0)
        {
            input.push_back(0);
            input.push_back(1);
            targets.push_back(1);
        }
        else if ((i+2) % 4 == 0)
        {
            input.push_back(1);
            input.push_back(0);
            targets.push_back(1);
        }
        else if ((i+3) % 4 == 0)
        {
            input.push_back(1);
            input.push_back(1);
            targets.push_back(0);
        }

        net.train(input, targets);
        net.results(output);

        std::cout << "Input: " << input[0] << " " << input[1] << '\n';
        std::cout << "Target: " << targets[0] << '\n';
        std::cout << "Output: " << output[0] << '\n';

        input.clear();
        targets.clear();
        output.clear();
    }
    */
}