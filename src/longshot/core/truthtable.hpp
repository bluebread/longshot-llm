#ifndef __TRUTHTABLE_HPP__
#define __TRUTHTABLE_HPP__

#include <random>
#include <limits>
#include <cstring>
#include <new>
#include <stdexcept>
#include <cstdlib>

#include "utils.hpp"

namespace longshot 
{
    class TruthTable
    {
    private:
        int num_vars_;
        size_t capacity_;
        uint64_t *chunks_;

    public:
        TruthTable(int n) : num_vars_(n), chunks_(nullptr)
        {
            if (n < 0 || n >= 64)
            {
                throw std::invalid_argument("TruthTable: n must be in [0, 63]");
            }
            capacity_ = (longshot::pow2(n) + 63) / 64 * sizeof(uint64_t);
            chunks_ = (uint64_t *)malloc(capacity_);
            if (chunks_ == nullptr)
            {
                throw std::bad_alloc();
            }
            memset(chunks_, 0, capacity_);
        }
        TruthTable(const TruthTable &other) : chunks_(nullptr)
        {
            capacity_ = other.capacity_;
            chunks_ = (uint64_t *)malloc(capacity_);
            if (chunks_ == nullptr)
            {
                throw std::bad_alloc();
            }
            memcpy(chunks_, other.chunks_, capacity_);
        }
        TruthTable(TruthTable &&other) : chunks_(nullptr)
        {
            capacity_ = other.capacity_;
            chunks_ = other.chunks_;
            other.chunks_ = nullptr;
            other.capacity_ = 0;
        }

        ~TruthTable()
        {
            free(chunks_);
        }

        void set()
        {
            memset(chunks_, 0xFF, capacity_);
        }
        void set(long long int x)
        {
            chunks_[x / 64] |= (1ull << (x % 64));
        }
        void reset()
        {
            memset(chunks_, 0, capacity_);
        }
        void reset(long long int x)
        {
            chunks_[x / 64] &= ~(1ull << (x % 64));
        }
        bool operator[](long long int x) const
        {
            return (chunks_[x / 64] >> (x % 64)) & 1;
        }
    };
}

#endif