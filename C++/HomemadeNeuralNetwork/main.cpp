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
    
    layer.units = 1;
    layer.layer_type = "Dense";
    layer.activation_function = "ReLu";
    topology.push_back(layer);

    Net net(input_size, topology);

    std::vector<double> input;
    std::vector<double> targets;
    std::vector<double> output;

    input.push_back(0);
    input.push_back(0);
    targets.push_back(0);

    net.predict(input, output);
    std::cout << "Output: " << output[0] << '\n';

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