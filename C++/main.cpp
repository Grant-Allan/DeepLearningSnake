#include "./NeuralNetworkLib/source/NeuralNetwork.cpp"
#include <iostream>
#include <fstream>
#include <sstream>


int main()
{
    std::vector<unsigned> topology;
    topology.push_back(2); // input (not an actual layer)
    topology.push_back(4); // hidden
    topology.push_back(1); // output
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