#include <iostream>

#define TESTER  "circuit"

#include "testsuite.hpp"
#include "circuit.hpp"

void test_circuit() {
    const int num_vars = 3;

    {
        longshot::NormalFormFormula<num_vars> formula;
    }
}

int main () {
    try
    {
        test_circuit();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        PRINT_FAILED;

        return 1;
    }
    
    PRINT_PASS;

    return 0;
}