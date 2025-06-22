#include "neuralnet.hpp"

#include <util/ezlog.hpp>

namespace mesa
{   // constructor
    NeuralNet::NeuralNet(const std::vector<uint32_t>& layout)
    {   // verify size
        ezlog::Logger::ASSERT(layout.size() >= 2, "Neural network cannot be constructed with less than 2 layers.");

        for (uint32_t i = 1; i < layout.size(); i++) // start one past input layer
        {   // layer
            m_Layers.emplace_back();
            Layer& layer = m_Layers[i - 1];
            layer.neurons.reserve(layout[i]);
            for (uint32_t n = 0; n < layer.neurons.capacity(); n++)
            {   // neuron
                layer.neurons.emplace_back();
                Neuron& neuron = layer.neurons[n];
                // weights
                neuron.Weights.reserve(layout[i - 1]);
                for (uint32_t w = 0; w < neuron.Weights.capacity(); w++) {
                    neuron.Weights.push_back(NeuralNet::RandomWeight());
                }
            }
        }
    }

    // public methods
    void NeuralNet::Propagate(const std::vector<double>& inputs)
    {
        sadf
    }

    void NeuralNet::BackwardPropagate(const std::vector<double>& target)
    {}

    // static methods
    double NeuralNet::RandomWeight()
    {   // returns a double between -0.1 and 0.1 for weight initialization
        return 0.1 * ((double)(rand()) / (double)(RAND_MAX / 2) - 1.0);
    }
}
