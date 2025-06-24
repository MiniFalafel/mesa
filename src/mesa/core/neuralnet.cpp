#include "neuralnet.hpp"

#include <util/ezlog.hpp>
#include <cmath>

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
                    neuron.Weights.push_back(0.5 * NeuralNet::RandomWeight() / std::sqrt(layer.Neurons.capacity()));
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
        ezlog::Logger::ASSERT(target.size() == m_Layers[m_Layers.size() - 1].Neurons.size(),
                "BACKPROPAGATION: target and output layer sizes do not match!");
        // loop backwards through layers
        for (uint32_t i = m_Layers.size() - 1; i > 0; i--) // excluding input layer
        {
            for (uint32_t n = 0; n < m_Layers[i].Neurons.size(); n++)
            {
                Neuron& neuron = m_Layers[i].Neurons[n];
                double error = 0.0;
                if (i == m_Layers.size() - 1) {
                    ezlog::Logger::LogINFO(std::format("T: {}, A: {}", target[n], neuron.Value));
                    error = target[n] - neuron.Value;
                    error = error * error; // MSE
                    ezlog::Logger::LogINFO(std::format("ERROR: {}", error));
                }
                else
                {   // accumulate from next layer
                    for (Neuron& nextNeuron : m_Layers[i + 1].Neurons)
                        error += nextNeuron.Weights[n] * nextNeuron.Delta;
                }
                // multiply by derivative
                neuron.Delta = error * neuron.Derivative();
            }
        }
    }

    void NeuralNet::UpdateWeights(const std::vector<double>& inputs)
    {
        std::vector<double> ins = inputs;
        for (uint32_t l = 0; l < m_Layers.size(); l++)
        {
            std::vector<double> nextIns;
            nextIns.reserve(m_Layers[l].Neurons.size());
            for (Neuron& neuron : m_Layers[l].Neurons)
            {
                ezlog::Logger::ASSERT(ins.size() == neuron.Weights.size(), "UpdateWeights: Last layer output, this layer weight mismatch occured.");
                for (uint32_t i = 0; i < neuron.Weights.size(); i++)
                    neuron.Weights[i] -= m_Settings.LearningRate * neuron.Delta * ins[i];
                // update bias
                neuron.Bias -= m_Settings.LearningRate * neuron.Delta;
                nextIns.push_back(neuron.Value);
            }
            ins = nextIns;
        }
    }

    // static methods
    double NeuralNet::RandomWeight()
    {   // returns a double between -0.1 and 0.1 for weight initialization
        return (double)(rand()) / (double)(RAND_MAX / 2) - 1.0;
    }
}
