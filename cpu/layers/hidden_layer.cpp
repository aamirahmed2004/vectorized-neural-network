#include "hidden_layer.hpp"
#include "../matrix_utils/activation.hpp"

HiddenLayer::HiddenLayer(const int numNeuronsCurrent, const int numNeuronsPrevious, const int batchSize, const Activation activationType)
    : m_inputMatrix(numNeuronsPrevious, batchSize),               // n_prev x batch size
      m_outputMatrix(numNeuronsCurrent, batchSize),               // n_current x batch size
      m_weightMatrix(numNeuronsCurrent, numNeuronsPrevious),      // n_current x n_prev
      m_gradients(numNeuronsCurrent, numNeuronsPrevious)          // n_current x n_prev
{
    m_numNeuronsCurrent = numNeuronsCurrent;
    m_numNeuronsPrevious = numNeuronsPrevious;
    m_activationType = activationType;
    m_batchSize = batchSize;
}
