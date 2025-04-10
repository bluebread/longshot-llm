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
        TESTCASE(1.5f, dnf.avgQ());
        
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
        TESTCASE(2.25f, dnf.avgQ());
        
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
        TESTCASE(2.0f, dnf.avgQ());

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
        TESTCASE(0.0f, dnf.avgQ());
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
        TESTCASE(1.5f, cnf.avgQ());
        
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
        TESTCASE(2.25f, cnf.avgQ());
        
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
        TESTCASE(2.0f, cnf.avgQ());
        
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
        TESTCASE(0.0f, cnf.avgQ());
    }
    {
        NormalFormFormula<3> xor3(NormalFormFormula<3>::Type::Disjunctive);
        
        xor3.add_clause({0b001, 0b110});
        xor3.add_clause({0b010, 0b101});
        xor3.add_clause({0b100, 0b011});
        xor3.add_clause({0b111, 0b000});
        TESTCASE(4, xor3.size());
        TESTCASE(3, xor3.width());
        TESTCASE(0, xor3.eval(0b000));
        TESTCASE(1, xor3.eval(0b001));
        TESTCASE(1, xor3.eval(0b010));
        TESTCASE(0, xor3.eval(0b011));
        TESTCASE(1, xor3.eval(0b100));
        TESTCASE(0, xor3.eval(0b101));
        TESTCASE(0, xor3.eval(0b110));
        TESTCASE(1, xor3.eval(0b111));
        TESTCASE(false, xor3.is_constant());
        TESTCASE(3.0f, xor3.avgQ());
        
        NormalFormFormula<4> xor4(NormalFormFormula<4>::Type::Disjunctive);
        
        xor4.add_clause({0b0001, 0b1110});
        xor4.add_clause({0b0010, 0b1101});
        xor4.add_clause({0b0100, 0b1011});
        xor4.add_clause({0b0111, 0b1000});
        xor4.add_clause({0b1000, 0b0111});
        xor4.add_clause({0b1011, 0b0100});
        xor4.add_clause({0b1101, 0b0010});
        xor4.add_clause({0b1110, 0b0001});
        TESTCASE(8, xor4.size());
        TESTCASE(4, xor4.width());
        TESTCASE(0, xor4.eval(0b0000));
        TESTCASE(1, xor4.eval(0b0001));
        TESTCASE(1, xor4.eval(0b0010));
        TESTCASE(0, xor4.eval(0b0011));
        TESTCASE(1, xor4.eval(0b0100));
        TESTCASE(0, xor4.eval(0b0101));
        TESTCASE(0, xor4.eval(0b0110));
        TESTCASE(1, xor4.eval(0b0111));
        TESTCASE(1, xor4.eval(0b1000));
        TESTCASE(0, xor4.eval(0b1001));
        TESTCASE(0, xor4.eval(0b1010));
        TESTCASE(1, xor4.eval(0b1011));
        TESTCASE(0, xor4.eval(0b1100));
        TESTCASE(1, xor4.eval(0b1101));
        TESTCASE(1, xor4.eval(0b1110));
        TESTCASE(0, xor4.eval(0b1111));
        TESTCASE(4.0f, xor4.avgQ());
        TESTCASE(false, xor4.is_constant());
        
        NormalFormFormula<5> xor5(NormalFormFormula<5>::Type::Disjunctive);
        
        xor5.add_clause({0b00001, 0b11110});
        xor5.add_clause({0b00010, 0b11101});
        xor5.add_clause({0b00100, 0b11011});
        xor5.add_clause({0b00111, 0b11000});
        xor5.add_clause({0b01000, 0b10111});
        xor5.add_clause({0b01011, 0b10100});
        xor5.add_clause({0b01101, 0b10010});
        xor5.add_clause({0b01110, 0b10001});
        xor5.add_clause({0b10000, 0b01111});
        xor5.add_clause({0b10011, 0b01100});
        xor5.add_clause({0b10101, 0b01010});
        xor5.add_clause({0b10110, 0b01001});
        xor5.add_clause({0b11001, 0b00110});
        xor5.add_clause({0b11010, 0b00101});
        xor5.add_clause({0b11100, 0b00011});
        xor5.add_clause({0b11111, 0b00000});
        TESTCASE(16, xor5.size());
        TESTCASE(5, xor5.width());
        TESTCASE(0, xor5.eval(0b00000));
        TESTCASE(1, xor5.eval(0b00001));
        TESTCASE(1, xor5.eval(0b00010));
        TESTCASE(0, xor5.eval(0b00011));
        TESTCASE(1, xor5.eval(0b00100));
        TESTCASE(0, xor5.eval(0b00101));
        TESTCASE(0, xor5.eval(0b00110));
        TESTCASE(1, xor5.eval(0b00111));
        TESTCASE(1, xor5.eval(0b01000));
        TESTCASE(0, xor5.eval(0b01001));
        TESTCASE(0, xor5.eval(0b01010));
        TESTCASE(1, xor5.eval(0b01011));
        TESTCASE(0, xor5.eval(0b01100));
        TESTCASE(1, xor5.eval(0b01101));
        TESTCASE(1, xor5.eval(0b01110));
        TESTCASE(0, xor5.eval(0b01111));
        TESTCASE(1, xor5.eval(0b10000));
        TESTCASE(0, xor5.eval(0b10001));
        TESTCASE(0, xor5.eval(0b10010));
        TESTCASE(1, xor5.eval(0b10011));
        TESTCASE(0, xor5.eval(0b10100));
        TESTCASE(1, xor5.eval(0b10101));
        TESTCASE(1, xor5.eval(0b10110));
        TESTCASE(0, xor5.eval(0b10111));
        TESTCASE(0, xor5.eval(0b11000));
        TESTCASE(1, xor5.eval(0b11001));
        TESTCASE(1, xor5.eval(0b11010));
        TESTCASE(0, xor5.eval(0b11011));
        TESTCASE(1, xor5.eval(0b11100));
        TESTCASE(0, xor5.eval(0b11101));
        TESTCASE(0, xor5.eval(0b11110));
        TESTCASE(1, xor5.eval(0b11111));
        TESTCASE(5.0f, xor5.avgQ());
        TESTCASE(false, xor5.is_constant());
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