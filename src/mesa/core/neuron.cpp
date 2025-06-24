#include "neuron.hpp"

#include <util/ezlog.hpp>
#include <cmath>

namespace mesa
{
    // methods
    double Neuron::Calculate(const std::vector<double>& inputs)
    {   // bias + dot product of inputs and weights
        ezlog::Logger::ASSERT(inputs.size() == Weights.size(), "inputs and weights are not of same size!");
        Value = Bias;
        for (uint32_t i = 0; i < Weights.size(); i++)
            Value += Weights[i] * inputs[i];
        // transfer function
        Value = Neuron::Transfer(TransferType, Value);

        return Value;
    }

    double Neuron::Derivative() const
    {
        return Neuron::d_Transfer(TransferType, Value);
    }

    // static methods
    double Neuron::Transfer(NeuronTransfer type, double x)
    {
        switch (type)
        {
            case NeuronTransfer::IDENTITY:
                return x;
            case NeuronTransfer::SIGMOID:
                return 1.0 / (1.0 + std::exp(-x));
            case NeuronTransfer::TANH:
                return std::tanh(x);
        }
    }

    double Neuron::d_Transfer(NeuronTransfer type, double tx)
    {
        switch (type)
        {
            case NeuronTransfer::IDENTITY:
                return 1.0;
            case NeuronTransfer::SIGMOID:
                // we do tx(1 - tx) because tx has already been through the sigmoid transfer func
                return tx * (1.0 - tx);
            case NeuronTransfer::TANH:
                // same here, tx = transfer(x)
                return 1.0 - tx * tx;
        }
    }
}
