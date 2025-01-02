#ifndef __LAYER_HPP
#define __LAYER_HPP

#include "../matrix_utils/matrix.hpp"
#include "../matrix_utils/matrix_lib.hpp"
#include "../matrix_utils/activation.hpp"

class HiddenLayer { 
private:
    Activation m_activationType;
    int m_numNeuronsCurrent;
    int m_numNeuronsPrevious;
    int m_batchSize;           

    /*
    One approach we could use for storing input and outputs of each layer is having 1D float vectors, i.e. each training example has its own input and output vector.
    Input vector would be numNeuronsPrevious elements long, and output vector would be numNeuronsCurrent elements long. 
    1 epoch would then be looping through each layer once for every training example in a batch, and the loop would have to store the gradients in a separate matrix. 

    The approach we take is using a 2D matrix-like structure, where the entire batch of training examples will have one input and output matrix.
    In the matrices, the number of rows is the number of neurons in the previous/current layer, and each column is the input/output for one training example. 
    1 epoch would then be looping through each layer once per batch.
    */

    Matrix &m_inputMatrix;         // Should be number of neurons in previous layer x number of training examples. For first hidden layer, it would be num features x batch size. 
    Matrix &m_weightMatrix;        // Should be number of neurons in current layer x number of neurons in previous layer 
    Matrix &m_gradients;           // Same dims as above, as each element in this matrix is the gradient descent update for the corresponding value in the weight matrix.
    Matrix &m_outputMatrix;        // Should be number of neurons in the current layer x number of training examples.

public:
    HiddenLayer(const int numNeuronsCurrent, const int numNeuronsPrevious, const int batchSize, const Activation activationType);
    
    void initializeWeights();
    Matrix getWeights();
    
    // Functions for gradient descent
    void linearForward();               // Part of forward pass; common for all hidden layers.
    virtual void applyActivation();     // Part of forward pass; will be specific to each subclass of HiddenLayer
    virtual void computeGradients();    // Backpropagating gradients; will be specific to each subclass of HiddenLayer
    void updateWeights(float alpha);    // Updating weights based on gradients matrix and learning rate = alpha.
};

#endif