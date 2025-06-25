#include "mesa/core/neuralnet.hpp"
#include <mesa/mesa.hpp>
#include <util/ezlog.hpp>

#include <iostream>

void displayVec(const char* label, std::vector<double> vec)
{
    std::cout << label << ": ";
    for (double& v : vec)
        std::cout << v << ", ";
    std::cout << "\n";
}

int main()
{
    //ezlog::Logger::LogINFO("Woah!");
    
    // create testing network
    std::vector<mesa::LayerSettings> layout = {
        { 2 },  // input
        { 2 },  // hidden
        { 1 }   // output
    };
    mesa::NeuralNet network(layout, { 0.5 });

    // check output
    std::vector<std::vector<double>> t_ins = {
        {0, 0}, {1, 0}, {1, 1}, {0, 1}
    };
    std::vector<std::vector<double>> t_outs = {
        {0}, {1}, {0}, {1}
    };
    std::cout << "BEFORE TRAINING:\n";
    for (uint32_t i = 0; i < t_ins.size(); i++)
    {
        std::vector<double> output = network.Propagate(t_ins[i]);
        displayVec("test input", t_ins[i]);
        displayVec("test out", output);
    }

    // train
    const uint32_t iters = 5000;
    for (uint32_t i = 0; i < iters; i++)
    {   // choose a random training option
        uint32_t trainIndex = rand() % t_ins.size();
        
        // train
        network.Propagate(t_ins[trainIndex]);
        network.BackwardPropagate(t_outs[trainIndex]);
        network.UpdateWeights(t_ins[trainIndex]);
    }

    // check output accuracy
    std::cout << "\nAFTER TRAINING:\n";
    for (uint32_t i = 0; i < t_ins.size(); i++)
    {
        std::vector<double> output = network.Propagate(t_ins[i]);
        displayVec("test input", t_ins[i]);
        displayVec("test out", output);
    }

}
