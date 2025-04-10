#ifndef __TEST_EXCEPTION_HPP__
#define __TEST_EXCEPTION_HPP__

#include <exception>
#include <string>
#include <sstream>
#include <iostream>
#include <new>
#include <cassert>

class TestException : public std::exception 
{
private:
    std::string message;

public:
    TestException(
            const char * filename, 
            const int lineno, 
            const char* testcase) {

        std::stringstream ss;

        ss << filename << ":" << lineno << ": " 
           <<  "\033[1;31m"   "error: "  "\033[0m"
           << "Testcase `" "\033[1;35m" << testcase << "\033[0m"  "` failed.";
        message = ss.str();
    }

    virtual const char* what() const throw() {
        return message.c_str();
    }
};

#define GREEN_PASS   "\033[1;32m"   "PASS"    "\033[0m"
#define RED_FAILED   "\033[1;31m"   "FAILED"  "\033[0m"

#ifdef TESTER
#define PRINT_PASS      std::cout << GREEN_PASS " - " TESTER << std::endl
#define PRINT_FAILED    std::cout << RED_FAILED " - " TESTER << std::endl
#else
#define PRINT_PASS      std::cout << GREEN_PASS << std::endl
#define PRINT_FAILED    std::cout << RED_FAILED << std::endl
#endif

#define TESTCASE(ANS, INPUT) \
    _testcase((ANS) == (INPUT), __FILE__, __LINE__, "(" #ANS ") == (" #INPUT ")")

void _testcase(bool condition, const char * filename, int lineno, const char * testcase_str)
{
    if (! condition) {
        throw TestException(filename, lineno, testcase_str);
    }
}

#endif