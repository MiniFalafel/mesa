#pragma once

#include <vector>

namespace mesa
{
    struct Neuron
    {
        std::vector<double> Weights; // TODO: allocate weight memory using custom array type that allocates within arena.
        double Value = 0.0;
        double Delta = 0.0;

        double Calculate(const std::vector<double>& inputs);
    };
}
