#include "neuron.hpp"

#include <util/ezlog.hpp>

namespace mesa
{
    double Neuron::Calculate(const std::vector<double>& inputs)
    {   // dot product of inputs and weights
        ezlog::Logger::ASSERT(inputs.size() == Weights.size(), "inputs and weights are not of same size!");
        Value = 0.0;
        for (uint32_t i = 0; i < Weights.size(); i++)
            Value += Weights[i] * inputs[i];

        return Value;
    }
}
