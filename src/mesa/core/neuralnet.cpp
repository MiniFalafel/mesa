#include "neuralnet.hpp"

#include <util/ezlog.hpp>

namespace mesa
{   // constructor
    NeuralNet::NeuralNet(const std::vector<LayerSettings>& layout, NeuralNetSettings settings)
        : m_Settings(settings)
    {   // verify size
        ezlog::Logger::ASSERT(layout.size() >= 2, "Neural network cannot be constructed with less than 2 layers.");

        for (uint32_t i = 1; i < layout.size(); i++) // start one past input layer
        {   // layer
            m_Layers.emplace_back();
            Layer& layer = m_Layers[i - 1];
            layer.Neurons.reserve(layout[i].Count);
            for (uint32_t n = 0; n < layer.Neurons.capacity(); n++)
            {   // neuron
                layer.Neurons.emplace_back();
                Neuron& neuron = layer.Neurons[n];
                neuron.TransferType = layout[i].Type;
                // weights
                neuron.Weights.reserve(layout[i - 1].Count);
                for (uint32_t w = 0; w < neuron.Weights.capacity(); w++) {
                    neuron.Weights.push_back(NeuralNet::RandomWeight());
                }
            }
        }
    }

    // public methods
    std::vector<double> NeuralNet::Propagate(const std::vector<double>& inputs)
    {
        std::vector<double> lastLayer = inputs;
        for (Layer& layer : m_Layers)
        {
            std::vector<double> thisLayer;
            thisLayer.reserve(layer.Neurons.size());
            for (Neuron& neuron : layer.Neurons)
            {
                neuron.Calculate(lastLayer);
                thisLayer.emplace_back(neuron.Value);
            }
            lastLayer = thisLayer;
        }
        return lastLayer;
    }

    void NeuralNet::BackwardPropagate(const std::vector<double>& target)
    {
        // TODO: Write this..
    }

    // static methods
    double NeuralNet::RandomWeight()
    {   // returns a double between -0.1 and 0.1 for weight initialization
        return 0.1 * ((double)(rand()) / (double)(RAND_MAX / 2) - 1.0);
    }
}
