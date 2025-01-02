#include <iostream>
#include <vector>

#include "layers/hidden_layer.hpp"
#include "matrix_utils/activation.hpp"

/*
    Reference for commands to compile and run executable from root directory: 
        g++ -o cpu/main cpu/main.cpp; cpu/main.exe
*/

int main(){
    HiddenLayer layer(5, Activation::ReLU);             // testing if the imports work
    std::cout << "Hello World!" << std::endl;
    return 0;
}

