#include <iostream>

#define TESTER  "circuit"

#include "testsuite.hpp"
#include "circuit.hpp"

using namespace longshot;

void test_circuit() {
    {
        NormalFormFormula<3> dnf(NormalFormFormula<3>::Type::Disjunctive);

        dnf.add_clause({0b001, 0b010}); // x0 and not x1
        TESTCASE(1, dnf.size());
        TESTCASE(3, dnf.num_vars());
        TESTCASE(2, dnf.depth());
        TESTCASE(2, dnf.width());
        TESTCASE(0, dnf.eval(0b000));
        TESTCASE(1, dnf.eval(0b001));
        TESTCASE(0, dnf.eval(0b010));
        TESTCASE(0, dnf.eval(0b011));
        TESTCASE(0, dnf.eval(0b100));
        TESTCASE(1, dnf.eval(0b101));
        TESTCASE(0, dnf.eval(0b110));
        TESTCASE(0, dnf.eval(0b111));
        
        dnf.add_clause({0b110, 0b001}); // (not x0) and x1 and x2
        TESTCASE(3, dnf.width());
        TESTCASE(2, dnf.size());
        TESTCASE(0, dnf.eval(0b000));
        TESTCASE(1, dnf.eval(0b001));
        TESTCASE(0, dnf.eval(0b010));
        TESTCASE(0, dnf.eval(0b011));
        TESTCASE(0, dnf.eval(0b100));
        TESTCASE(1, dnf.eval(0b101));
        TESTCASE(1, dnf.eval(0b110));
        TESTCASE(0, dnf.eval(0b111));
        
        dnf.add_clause({0b000, 0b100}); // not x2
        TESTCASE(3, dnf.size());
        TESTCASE(1, dnf.eval(0b000));
        TESTCASE(1, dnf.eval(0b001));
        TESTCASE(1, dnf.eval(0b010));
        TESTCASE(1, dnf.eval(0b011));
        TESTCASE(0, dnf.eval(0b100));
        TESTCASE(1, dnf.eval(0b101));
        TESTCASE(1, dnf.eval(0b110));
        TESTCASE(0, dnf.eval(0b111));
        TESTCASE(false, dnf.is_constant());

        dnf.add_clause({0b100, 0b000}); // x2
        TESTCASE(4, dnf.size());
        TESTCASE(1, dnf.eval(0b000));
        TESTCASE(1, dnf.eval(0b001));
        TESTCASE(1, dnf.eval(0b010));
        TESTCASE(1, dnf.eval(0b011));
        TESTCASE(1, dnf.eval(0b100));
        TESTCASE(1, dnf.eval(0b101));
        TESTCASE(1, dnf.eval(0b110));
        TESTCASE(1, dnf.eval(0b111));
        TESTCASE(true, dnf.is_constant());
    }

    {
        NormalFormFormula<3> cnf(NormalFormFormula<3>::Type::Conjunctive);

        cnf.add_clause({0b001, 0b010}); // x0 or not x1
        TESTCASE(1, cnf.size());
        TESTCASE(3, cnf.num_vars());
        TESTCASE(2, cnf.depth());
        TESTCASE(2, cnf.width());
        TESTCASE(1, cnf.eval(0b000));
        TESTCASE(1, cnf.eval(0b001));
        TESTCASE(0, cnf.eval(0b010));
        TESTCASE(1, cnf.eval(0b011));
        TESTCASE(1, cnf.eval(0b100));
        TESTCASE(1, cnf.eval(0b101));
        TESTCASE(0, cnf.eval(0b110));
        TESTCASE(1, cnf.eval(0b111));
        TESTCASE(false, cnf.is_constant());
        
        cnf.add_clause({0b110, 0b001}); // (not x0) or x1 or x2
        TESTCASE(3, cnf.width());
        TESTCASE(2, cnf.size());
        TESTCASE(1, cnf.eval(0b000));
        TESTCASE(0, cnf.eval(0b001));
        TESTCASE(0, cnf.eval(0b010));
        TESTCASE(1, cnf.eval(0b011));
        TESTCASE(1, cnf.eval(0b100));
        TESTCASE(1, cnf.eval(0b101));
        TESTCASE(0, cnf.eval(0b110));
        TESTCASE(1, cnf.eval(0b111));
        
        cnf.add_clause({0b000, 0b100}); // not x2
        TESTCASE(3, cnf.size());
        TESTCASE(1, cnf.eval(0b000));
        TESTCASE(0, cnf.eval(0b001));
        TESTCASE(0, cnf.eval(0b010));
        TESTCASE(1, cnf.eval(0b011));
        TESTCASE(0, cnf.eval(0b100));
        TESTCASE(0, cnf.eval(0b101));
        TESTCASE(0, cnf.eval(0b110));
        TESTCASE(0, cnf.eval(0b111));
        TESTCASE(false, cnf.is_constant());
        
        cnf.add_clause({0b100, 0b000}); // x2
        TESTCASE(4, cnf.size());
        TESTCASE(0, cnf.eval(0b000));
        TESTCASE(0, cnf.eval(0b001));
        TESTCASE(0, cnf.eval(0b010));
        TESTCASE(0, cnf.eval(0b011));
        TESTCASE(0, cnf.eval(0b100));
        TESTCASE(0, cnf.eval(0b101));
        TESTCASE(0, cnf.eval(0b110));
        TESTCASE(0, cnf.eval(0b111));
        TESTCASE(true, cnf.is_constant());
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