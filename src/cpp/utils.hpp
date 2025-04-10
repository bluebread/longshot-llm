#ifndef __UTILS_HPP__
#define __UTILS_HPP__


namespace longshot
{
    constexpr unsigned long long int pow2(unsigned int n)
    {
        return (1ull << n);
    }

    constexpr unsigned long long int pow(unsigned int base, unsigned int exp)
    {
        return (exp == 0) ? 1 : base * pow(base, exp - 1);
    }

};

#endif // __UTILS_HPP__