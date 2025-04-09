#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <bitset>

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

    template<int num_vars_>
    struct _rstr_t 
    {
    protected:
        static_assert(num_vars_ <= 32, "_rstr_t: Number of variables must be less than or equal to 32.");
        std::bitset<num_vars_> vals_;
        std::bitset<num_vars_> unfixed_;
    public:
        _rstr_t() : vals_(0), unfixed_(0) {}
        _rstr_t(uint64_t num_repr) : vals_(0), unfixed_(0) 
        {
            if (num_repr >= pow(3, num_vars_)) {
                throw std::out_of_range("_rstr_t: `num_repr` exceeds the maximum value for the given number of variables.");
            }

            int count = 0;
            while (num_repr > 0)
            {
                switch (num_repr % 3)
                {
                case 0:
                    vals_.reset(count);
                    unfixed_.reset(count);
                    break;
                case 1:
                    vals_.set(count);
                    unfixed_.reset(count);
                    break;
                case 2:
                    vals_.reset(count);
                    unfixed_.set(count);
                    break;
                default:
                }

                num_repr /= 3;
                count++;
            }
        }

        const std::bitset<num_vars_> & vals() const { return vals_; }
        const std::bitset<num_vars_> & unfixed() const { return unfixed_; }

        bool all_fixed() const
        {
            return unfixed_.none();
        }

        std::vector<int> unfixed_bits() const
        {
            std::vector<int> res;
            for (int i = 0; i < num_vars_; i++)
            {
                if (unfixed_.test(i))
                    res.push_back(i);
            }
            return res;
        }
        
        _rstr_t fix(int idx, bool b) const
        {
            if (idx < 0 || idx >= num_vars_) {
                throw std::out_of_range("_rstr_t: Index out of range.");
            }

            _rstr_t cpy(*this);
            cpy.vals_[idx] = b;
            cpy.unfixed_.reset(idx);

            return cpy;
        }

        void next()
        {
            for (int i = 0; i < num_vars_; i++)
            {
                if (unfixed_.test(i))
                {
                    vals_.reset(i);
                    unfixed_.reset(i);
                }
                else 
                {
                    if (vals_[i] == 0)
                    {
                        vals_.set(i);
                        // unfixed_.reset(i);
                    }
                    else
                    {
                        vals_.reset(i);
                        unfixed_.set(i);
                    }
                    break;

                }
            }
        }
    };
};

#endif // __UTILS_HPP__