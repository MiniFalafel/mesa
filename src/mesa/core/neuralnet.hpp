#pragma once

#include "neuron.hpp"

#include <vector>
#include <stdint.h>

namespace mesa
{
    // layer
    struct Layer
    {
        std::vector<Neuron> Neurons; // TODO: custom vector type
    };

    struct LayerSettings
    {
        uint32_t Count;
        NeuronTransfer Type = NeuronTransfer::SIGMOID;
    };

    // neural net
    struct NeuralNetSettings
    {
        double LearningRate;
    };

    class NeuralNet
    {
        std::vector<Layer> m_Layers; // TODO: custom vector type
        NeuralNetSettings m_Settings;

    public:
        NeuralNet(const std::vector<LayerSettings>& layout, NeuralNetSettings settings = { 0.1 });
        // TODO: Create a constructor that loads serialized weights

        std::vector<double> Propagate(const std::vector<double>& inputs);
        void BackwardPropagate(const std::vector<double>& target);
        void UpdateWeights(const std::vector<double>& inputs);

    private:
        // static
        static double RandomWeight();
    };
}
