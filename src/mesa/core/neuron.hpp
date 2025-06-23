#pragma once

#include <vector>

namespace mesa
{
    // transfer functions
    enum class NeuronTransfer
    {
        IDENTITY,
        SIGMOID,
        TANH
    };

    struct Neuron
    {
        std::vector<double> Weights; // TODO: allocate weight memory using custom array type that allocates within arena.
        double Value = 0.0;
        double Delta = 0.0;
        NeuronTransfer TransferType = NeuronTransfer::SIGMOID;

        double Calculate(const std::vector<double>& inputs);

    private:
        static double Transfer(NeuronTransfer type, double x);
        static double d_Transfer(NeuronTransfer type, double x);
    };
}
