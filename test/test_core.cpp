#include <iostream>

#define TESTER  __FILE__

#include "testsuite.hpp"
#include "bool.hpp"
// #include "circuit.hpp"

using namespace longshot;

void test_bool() {
    {
        MonotonicBooleanFunction f(3);
        TESTCASE(3, f.num_vars());

        f.reset();
        TESTCASE(0b000, f.eval(0b000));
        TESTCASE(0b000, f.eval(0b001));
        TESTCASE(0b000, f.eval(0b010));
        TESTCASE(0b000, f.eval(0b011)); 
        TESTCASE(0b000, f.eval(0b100));
        TESTCASE(0b000, f.eval(0b101));
        TESTCASE(0b000, f.eval(0b110));
        TESTCASE(0b000, f.eval(0b111));
        
        Literals l1(0b001, 0b010);
        f.add_term(l1); // x0 and not x1
        TESTCASE(0, f.eval(0b000));
        TESTCASE(1, f.eval(0b001));
        TESTCASE(0, f.eval(0b010));
        TESTCASE(0, f.eval(0b011));
        TESTCASE(0, f.eval(0b100));
        TESTCASE(1, f.eval(0b101));
        TESTCASE(0, f.eval(0b110));
        TESTCASE(0, f.eval(0b111));
        TESTCASE(1.5f, f.avgQ());

        Literals l2(0b100, 0b110);
        f.add_term({0b110, 0b001}); // (not x0) and x1 and x2
        TESTCASE(0, f.eval(0b000));
        TESTCASE(1, f.eval(0b001));
        TESTCASE(0, f.eval(0b010));
        TESTCASE(0, f.eval(0b011));
        TESTCASE(0, f.eval(0b100));
        TESTCASE(1, f.eval(0b101));
        TESTCASE(1, f.eval(0b110));
        TESTCASE(0, f.eval(0b111));
        TESTCASE(2.25f, f.avgQ());
        
        Literals l3(0b100, 0b100);
        f.add_term(l3); // (not x0) and x1 and x2
        TESTCASE(2.25f, f.avgQ());

        Literals l4(0b000, 0b100);
        f.add_term(l4); // not x2
        TESTCASE(1, f.eval(0b000));
        TESTCASE(1, f.eval(0b001));
        TESTCASE(1, f.eval(0b010));
        TESTCASE(1, f.eval(0b011));
        TESTCASE(0, f.eval(0b100));
        TESTCASE(1, f.eval(0b101));
        TESTCASE(1, f.eval(0b110));
        TESTCASE(0, f.eval(0b111));
        TESTCASE(2.0f, f.avgQ());

        Literals l5(0b100, 0b000);
        f.add_term({0b100, 0b000}); // x2
        TESTCASE(1, f.eval(0b000));
        TESTCASE(1, f.eval(0b001));
        TESTCASE(1, f.eval(0b010));
        TESTCASE(1, f.eval(0b011));
        TESTCASE(1, f.eval(0b100));
        TESTCASE(1, f.eval(0b101));
        TESTCASE(1, f.eval(0b110));
        TESTCASE(1, f.eval(0b111));
        TESTCASE(0.0f, f.avgQ());
    }
    {
        CountingBooleanFunction f(3);
        TESTCASE(3, f.num_vars());
        
        f.set();
        TESTCASE(1, f.eval(0b000));
        TESTCASE(1, f.eval(0b001));
        TESTCASE(1, f.eval(0b010));
        TESTCASE(1, f.eval(0b011));
        TESTCASE(1, f.eval(0b100));
        TESTCASE(1, f.eval(0b101));
        TESTCASE(1, f.eval(0b110));
        TESTCASE(1, f.eval(0b111));
        
        Literals l1(0b001, 0b010);
        f.add_clause(l1); // x0 or not x1
        TESTCASE(1, f.eval(0b000));
        TESTCASE(1, f.eval(0b001));
        TESTCASE(0, f.eval(0b010));
        TESTCASE(1, f.eval(0b011));
        TESTCASE(1, f.eval(0b100));
        TESTCASE(1, f.eval(0b101));
        TESTCASE(0, f.eval(0b110));
        TESTCASE(1, f.eval(0b111));
        TESTCASE(1.5f, f.avgQ());
        
        Literals l2(0b100, 0b100);
        f.add_clause(l2); // x0 or not x1
        TESTCASE(1.5f, f.avgQ());
        
        Literals l3(0b110, 0b001);
        f.add_clause(l3); // (not x0) or x1 or x2
        TESTCASE(1, f.eval(0b000));
        TESTCASE(0, f.eval(0b001));
        TESTCASE(0, f.eval(0b010));
        TESTCASE(1, f.eval(0b011));
        TESTCASE(1, f.eval(0b100));
        TESTCASE(1, f.eval(0b101));
        TESTCASE(0, f.eval(0b110));
        TESTCASE(1, f.eval(0b111));
        TESTCASE(2.25f, f.avgQ());
        
        Literals l4(0b000, 0b100);
        f.add_clause(l4); // not x2
        TESTCASE(1, f.eval(0b000));
        TESTCASE(0, f.eval(0b001));
        TESTCASE(0, f.eval(0b010));
        TESTCASE(1, f.eval(0b011));
        TESTCASE(0, f.eval(0b100));
        TESTCASE(0, f.eval(0b101));
        TESTCASE(0, f.eval(0b110));
        TESTCASE(0, f.eval(0b111));
        TESTCASE(2.0f, f.avgQ());
        
        Literals l5(0b100, 0b000);
        f.add_clause(l5); // x2
        TESTCASE(0, f.eval(0b000));
        TESTCASE(0, f.eval(0b001));
        TESTCASE(0, f.eval(0b010));
        TESTCASE(0, f.eval(0b011));
        TESTCASE(0, f.eval(0b100));
        TESTCASE(0, f.eval(0b101));
        TESTCASE(0, f.eval(0b110));
        TESTCASE(0, f.eval(0b111));
        TESTCASE(0.0f, f.avgQ());

        f.del_clause(l5); // x2
        TESTCASE(1, f.eval(0b000));
        TESTCASE(0, f.eval(0b001));
        TESTCASE(0, f.eval(0b010));
        TESTCASE(1, f.eval(0b011));
        TESTCASE(0, f.eval(0b100));
        TESTCASE(0, f.eval(0b101));
        TESTCASE(0, f.eval(0b110));
        TESTCASE(0, f.eval(0b111));
        TESTCASE(2.0f, f.avgQ());
        
        f.del_clause(l4); // not x2
        TESTCASE(1, f.eval(0b000));
        TESTCASE(0, f.eval(0b001));
        TESTCASE(0, f.eval(0b010));
        TESTCASE(1, f.eval(0b011));
        TESTCASE(1, f.eval(0b100));
        TESTCASE(1, f.eval(0b101));
        TESTCASE(0, f.eval(0b110));
        TESTCASE(1, f.eval(0b111));
        TESTCASE(2.25f, f.avgQ());
        
        f.del_clause(l3); // (not x0) and x1 and x2
        TESTCASE(1.5f, f.avgQ());
        
        f.del_clause(l2); // (not x0) and x1 and x2
        TESTCASE(1, f.eval(0b000));
        TESTCASE(1, f.eval(0b001));
        TESTCASE(0, f.eval(0b010));
        TESTCASE(1, f.eval(0b011));
        TESTCASE(1, f.eval(0b100));
        TESTCASE(1, f.eval(0b101));
        TESTCASE(0, f.eval(0b110));
        TESTCASE(1, f.eval(0b111));
        TESTCASE(1.5f, f.avgQ());
        
        f.del_clause(l1); // x0 and not x1
        TESTCASE(1, f.eval(0b000));
        TESTCASE(1, f.eval(0b001));
        TESTCASE(1, f.eval(0b010));
        TESTCASE(1, f.eval(0b011));
        TESTCASE(1, f.eval(0b100));
        TESTCASE(1, f.eval(0b101));
        TESTCASE(1, f.eval(0b110));
        TESTCASE(1, f.eval(0b111));
    }
    {
        CountingBooleanFunction xor4(4);
        
        Literals l1(0b0001, 0b1110);
        Literals l2(0b0010, 0b1101);
        Literals l3(0b0100, 0b1011);
        Literals l4(0b0111, 0b1000);
        Literals l5(0b1000, 0b0111);
        Literals l6(0b1011, 0b0100);
        Literals l7(0b1101, 0b0010);
        Literals l8(0b1110, 0b0001);

        xor4.add_term(l1);
        xor4.add_term(l2);
        xor4.add_term(l3);
        xor4.add_term(l4);
        xor4.add_term(l5);
        xor4.add_term(l6);
        xor4.add_term(l7);
        xor4.add_term(l8);

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

        xor4.del_term(l8);
        xor4.del_term(l3);
        xor4.del_term(l4);
        xor4.del_term(l5);
        xor4.del_term(l6);
        xor4.del_term(l7);
        xor4.del_term(l2);
        xor4.del_term(l1);

        TESTCASE(0, xor4.eval(0b0000));
        TESTCASE(0, xor4.eval(0b0001));
        TESTCASE(0, xor4.eval(0b0010));
        TESTCASE(0, xor4.eval(0b0011));
        TESTCASE(0, xor4.eval(0b0100));
        TESTCASE(0, xor4.eval(0b0101));
        TESTCASE(0, xor4.eval(0b0110));
        TESTCASE(0, xor4.eval(0b0111));
        TESTCASE(0, xor4.eval(0b1000));
        TESTCASE(0, xor4.eval(0b1001));
        TESTCASE(0, xor4.eval(0b1010));
        TESTCASE(0, xor4.eval(0b1011));
        TESTCASE(0, xor4.eval(0b1100));
        TESTCASE(0, xor4.eval(0b1101));
        TESTCASE(0, xor4.eval(0b1110));
        TESTCASE(0, xor4.eval(0b1111));
        TESTCASE(0.0f, xor4.avgQ());
    }
    // TODO: more tests
}

int main () {
    try
    {
        // test_circuit();
        test_bool();
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