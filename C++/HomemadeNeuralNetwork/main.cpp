#include "NeuralNet.cpp"

int main()
{
    // Topology, learning rate
    std::vector<unsigned> topology = {2, 3, 1};
    float learning_rate = 0.001f;
    NeuralNetwork net(topology, learning_rate);

    RowVector input(2);
    RowVector targets(1);
    RowVector results(1);


    // Train the model to be an XOR gate
    for (int i = 0; i < 2000; i++)
    {
        std::cout << '\n' << "Episode: " << i << '\n';

        if (i % 4 == 0)
        {
            input[0] = 0;
            input[1] = 0;
            results[0] = 0;
        }
        else if ((i+1) % 4 == 0)
        {
            input[0] = 0;
            input[1] = 1;
            results[0] = 1;
        }
        else if ((i+2) % 4 == 0)
        {
            input[0] = 1;
            input[1] = 0;
            results[0] = 1;
        }
        else if ((i+3) % 4 == 0)
        {
            input[0] = 1;
            input[1] = 1;
            results[0] = 0;
        }

        net.train(input, results);

        std::cout << "Input: " << input[0] << " " << input[1] << '\n';
        std::cout << "Target: " << targets[0] << '\n';
        std::cout << "Results: " << results[0] << '\n';
    }

    return 0;
}