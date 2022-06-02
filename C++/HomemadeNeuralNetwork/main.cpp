#include "./NeuralNetLib/source/NeuralNet.cpp"
#include <iostream>
#include <fstream>
#include <sstream>


int main()
{
    std::vector<LayerSet> topology;
    LayerSet layer;
    
    layer.neurons = 2;
    layer.layer_type = "Dense";
    layer.activation_function = "ReLu";
    topology.push_back(layer); // input (not an actual layer)

    layer.neurons = 4;
    layer.layer_type = "Dense";
    layer.activation_function = "ReLu";
    topology.push_back(layer); // hidden

    layer.neurons = 1;
    layer.layer_type = "Dense";
    layer.activation_function = "ReLu";
    topology.push_back(layer); // output
    Net net(topology);


    std::vector<double> input;
    std::vector<double> targets;
    std::vector<double> results;


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

        net.feedForward(input);
        net.backPropagation(targets);
        net.getResults(results);

        std::cout << "Input: " << input[0] << " " << input[1] << '\n';
        std::cout << "Target: " << targets[0] << '\n';
        std::cout << "Results: " << results[0] << '\n';

        input.clear();
        targets.clear();
        results.clear();
    }

    return 0;
}