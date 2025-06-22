#pragma once

#include "neuron.hpp"

#include <vector>
#include <stdint.h>

namespace mesa
{
    struct Layer
    {
        std::vector<Neuron> neurons; // TODO: custom vector type
    };

    class NeuralNet
    {
        std::vector<Layer> m_Layers; // TODO: custom vector type

    public:
        NeuralNet(const std::vector<uint32_t>& layout);
        // TODO: Create a constructor that loads serialized weights

        void Propagate(const std::vector<double>& inputs);
        void BackwardPropagate(const std::vector<double>& target);

    private:
        static double RandomWeight();
    };
}
