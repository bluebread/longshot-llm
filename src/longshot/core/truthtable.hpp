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
    class SimpleTruthTable
    {
    private:
        int num_vars_;
        size_t capacity_;
        uint64_t *chunks_;

    public:
        SimpleTruthTable(int n) : num_vars_(n), chunks_(nullptr)
        {
            if (n < 0 || n >= 64)
            {
                throw std::invalid_argument("SimpleTruthTable: n must be in [0, 63]");
            }
            capacity_ = (longshot::pow2(n) + 63) / 64 * sizeof(uint64_t);
            chunks_ = (uint64_t *)malloc(capacity_);
            if (chunks_ == nullptr)
            {
                throw std::bad_alloc();
            }
            memset(chunks_, 0, capacity_);
        }
        SimpleTruthTable(const SimpleTruthTable &other) : chunks_(nullptr)
        {
            num_vars_ = other.num_vars_;
            capacity_ = other.capacity_;
            chunks_ = (uint64_t *)malloc(capacity_);
            if (chunks_ == nullptr)
            {
                throw std::bad_alloc();
            }
            memcpy(chunks_, other.chunks_, capacity_);
        }
        SimpleTruthTable(SimpleTruthTable &&other) : chunks_(nullptr)
        {
            capacity_ = other.capacity_;
            chunks_ = other.chunks_;
            other.chunks_ = nullptr;
            other.capacity_ = 0;
        }

        ~SimpleTruthTable()
        {
            free(chunks_);
        }

        void set()
        {
            memset(chunks_, 0xFF, capacity_);
        }
        void set(uint32_t x)
        {
            chunks_[x / 64] |= (1ull << (x % 64));
        }
        void reset()
        {
            memset(chunks_, 0, capacity_);
        }
        void reset(uint32_t x)
        {
            chunks_[x / 64] &= ~(1ull << (x % 64));
        }
        bool operator[](uint32_t x) const
        {
            return (chunks_[x / 64] >> (x % 64)) & 1;
        }
    };

    class CountingTruthTable
    {
    private:
        int num_vars_;
        uint32_t *chunks_;
    public:
        CountingTruthTable(int n) : num_vars_(n), chunks_(nullptr)
        {
            if (n < 0 || n >= 32)
            {
                throw std::invalid_argument("CountingTruthTable: n must be in [0, 31]");
            }
            chunks_ = (uint32_t *)malloc(sizeof(uint32_t) * longshot::pow2(n));
            if (chunks_ == nullptr)
            {
                throw std::bad_alloc();
            }
            memset(chunks_, 0, sizeof(uint32_t) * longshot::pow2(n));
        }
        CountingTruthTable(const CountingTruthTable &other) : num_vars_(other.num_vars_), chunks_(nullptr)
        {
            chunks_ = (uint32_t *)malloc(sizeof(uint32_t) * longshot::pow2(num_vars_));
            if (chunks_ == nullptr)
            {
                throw std::bad_alloc();
            }
            memcpy(chunks_, other.chunks_, sizeof(uint32_t) * longshot::pow2(num_vars_));
        }
        CountingTruthTable(CountingTruthTable &&other) : num_vars_(other.num_vars_), chunks_(nullptr)
        {
            chunks_ = other.chunks_;
            other.chunks_ = nullptr;
        }
        ~CountingTruthTable()
        {
            free(chunks_);
        }

        void reset()
        {
            memset(chunks_, 0, sizeof(uint32_t) * longshot::pow2(num_vars_));
        }
        void inc(uint32_t x)
        {
            chunks_[x] += 1;
        }
        void dec(uint32_t x)
        {
            chunks_[x] -= 1;
        }
        bool operator[](uint32_t x) const
        {
            return chunks_[x] > 0;
        }
        uint32_t count(uint32_t x) const
        {
            return chunks_[x];
        }

    };
}

#endif