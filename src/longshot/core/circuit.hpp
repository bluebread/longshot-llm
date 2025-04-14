#ifndef __CIRCUIT_HPP__
#define __CIRCUIT_HPP__

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <algorithm>
#include <bitset>
#include <iostream>

#include "utils.hpp"

namespace longshot
{
    class AC0_Circuit
    {
    protected:
        int size_ = 0;
        const int num_vars_;
        const int depth_;

    public:
        typedef uint32_t input_t;

        AC0_Circuit(int n, int d) : num_vars_(n), depth_(d) {}
        ~AC0_Circuit() {}

        int num_vars() const { return num_vars_; }
        int size() const { return size_; }
        int depth() const { return depth_; }

        virtual bool eval(input_t x) const = 0;
        virtual double avgQ() const = 0;

        virtual bool is_constant() const
        {
            if (size_ == 0)
            {
                return true;
            }

            bool v0 = eval(0);
            uint64_t num_inputs = longshot::pow2(num_vars_);

            for (uint64_t i = 1; i < num_inputs; i++)
            {
                if (eval(i) != v0)
                {
                    return false;
                }
            }
            return true;
        }
    };

    class NormalFormFormula : public AC0_Circuit
    {
        using input_t = typename AC0_Circuit::input_t;

    public:
        enum class Type
        {
            Conjunctive, // Conjunctive Normal Form
            Disjunctive, // Disjunctive Normal Form
        };

        struct Clause
        {
            input_t pos_vars;
            input_t neg_vars;

            Clause(input_t vars, input_t neg_vars) : pos_vars(vars), neg_vars(neg_vars) {}
            Clause(const Clause &other) : pos_vars(other.pos_vars), neg_vars(other.neg_vars) {}
            Clause() : pos_vars(0), neg_vars(0) {}
        };

    private:
        struct _trtb_t
        {
        private:
            uint64_t *chunks_;
            size_t capacity_;

        public:
            _trtb_t(int n) : chunks_(nullptr)
            {
                capacity_ = (longshot::pow2(n) + 63) / 64 * sizeof(uint64_t);
                chunks_ = (uint64_t *)malloc(capacity_);
                if (chunks_ == nullptr)
                {
                    throw std::bad_alloc();
                }
                memset(chunks_, 0, capacity_);
            }
            _trtb_t(const _trtb_t &other) : chunks_(nullptr)
            {
                capacity_ = other.capacity_;
                chunks_ = (uint64_t *)malloc(capacity_);
                if (chunks_ == nullptr)
                {
                    throw std::bad_alloc();
                }
                memcpy(chunks_, other.chunks_, capacity_);
            }
            _trtb_t(_trtb_t &&other) : chunks_(nullptr)
            {
                capacity_ = other.capacity_;
                chunks_ = other.chunks_;
                other.chunks_ = nullptr;
                other.capacity_ = 0;
            }

            ~_trtb_t()
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

    protected:
        const Type type_ = Type::Disjunctive;

        int width_ = 0;
        std::vector<Clause> clauses_;
        _trtb_t truth_table_;

    public:
        NormalFormFormula(int n, Type type = Type::Disjunctive) : 
            AC0_Circuit(n, 2), type_(type), truth_table_(n)
        {
            if (type_ == Type::Disjunctive)
            {
                truth_table_.reset();
            }
            else
            {
                truth_table_.set();
            }
        }
        NormalFormFormula(const NormalFormFormula &other) : 
            AC0_Circuit(other.num_vars_, other.depth_), type_(other.type_), width_(other.width_), truth_table_(other.truth_table_)
        {
            clauses_ = other.clauses_;
        }
        NormalFormFormula(NormalFormFormula &&other) : 
            AC0_Circuit(other.num_vars_, other.depth_), type_(other.type_), width_(other.width_), truth_table_(other.truth_table_)
        {
            clauses_ = std::move(other.clauses_);
        }

        ~NormalFormFormula() {}

        int width() const { return width_; }
        const std::vector<Clause> & clauses() const { return clauses_; }

        void add_clause(Clause cl)
        {
            int num_pv = __builtin_popcount(cl.pos_vars);
            int num_nv = __builtin_popcount(cl.neg_vars);

            if (num_pv == 0 && num_nv == 0)
            {
                return;
            }

            clauses_.push_back(cl);
            this->size_ += 1;
            int cl_width = __builtin_popcount(cl.pos_vars | cl.neg_vars);
            width_ = std::max(width_, cl_width);

            input_t mask = cl.pos_vars | cl.neg_vars;
            input_t x = 0;

            for (uint64_t i = 0; i < longshot::pow2(num_vars_ - cl_width); i++)
            {
                if (type_ == Type::Disjunctive)
                {
                    input_t y = (x | cl.pos_vars) & ~cl.neg_vars;
                    truth_table_.set(y);
                }
                else
                {
                    input_t y = (x | cl.neg_vars) & ~cl.pos_vars;
                    truth_table_.reset(y);
                }

                // magic bit manipulation: enumerate all inputs that satisfies the clause
                input_t z = (mask | x) + 1;
                z = z & -z;
                x = (~(z - 1) & x) | z;
            }
        }

        bool eval(input_t x) const
        {
            return truth_table_[x];
        }

    private:
        union _dp_item_t
        {
            struct
            {
                bool val : 1;        // valid if depth_numerator == 0
                uint32_t depth : 31; // represented by an integer due to D_ave(f) * 2^n
            };

            uint32_t raw;

            _dp_item_t() : raw(0) {}
            _dp_item_t(bool v, uint32_t dn) : raw(0)
            {
                this->val = v;
                this->depth = dn;
            }
        };

        struct _rstr_t
        {
        protected:
            input_t vals_;
            input_t unfixed_;

        public:
            _rstr_t() : vals_(0), unfixed_(0) {}
            _rstr_t(const _rstr_t &r) : vals_(r.vals_), unfixed_(r.unfixed_) {}

            const input_t &vals() const { return vals_; }
            const input_t &unfixed() const { return unfixed_; }

            bool all_fixed() const
            {
                return unfixed_ == 0;
            }

            _rstr_t fix(int idx, bool b) const
            {
                _rstr_t cpy(*this);
                cpy.vals_ = (cpy.vals_ & ~(1u << idx)) | ((b & 1u) << idx);
                cpy.unfixed_ &= ~(1u << idx);

                return cpy;
            }

            void next()
            {
                unfixed_ += 1;
                input_t x = unfixed_ & -unfixed_; // get the rightmost 1 bit
                unfixed_ ^= (x ^ (x & vals_));
                vals_ ^= x;
            }
        };

        std::string _tris_str(uint64_t num_repr) const
        {
            std::string s = "";
            for (int i = 0; i < num_vars_; i++)
            {
                switch (num_repr % 3)
                {
                case 0:
                    s += "0";
                    break;
                case 1:
                    s += "1";
                    break;
                case 2:
                    s += "?";
                    break;
                default:
                    throw std::logic_error("_rstr_t: Invalid representation.");
                }

                num_repr /= 3;
            }
            std::reverse(s.begin(), s.end());

            return s;
        }

    public:
        double avgQ() const
        {
            uint64_t *exp3_tb = (uint64_t *)alloca(num_vars_ * sizeof(uint64_t));

            exp3_tb[0] = 1;

            for (int i = 1; i < num_vars_; i++)
            {
                exp3_tb[i] = 3 * exp3_tb[i - 1];
            }

            uint64_t exp3_n = 3 * exp3_tb[num_vars_ - 1];
            _dp_item_t *lookup = new _dp_item_t[exp3_n];
            _rstr_t rt;

            for (uint64_t i = 0; i < exp3_n; i++)
            {

                if (rt.all_fixed())
                {
                    bool v = eval(rt.vals());
                    lookup[i] = _dp_item_t(v, 0);
                    rt.next();
                    continue;
                }

                uint32_t min_d = std::numeric_limits<uint32_t>::max();
                _dp_item_t choice;
                uint32_t x = rt.unfixed();
                unsigned int num_unfixed = __builtin_popcount(x);

                while (x != 0)
                {
                    unsigned int mask = x & -x;
                    unsigned int b = __builtin_ctz(mask);

                    _dp_item_t sf0 = lookup[i - 2 * exp3_tb[b]];
                    _dp_item_t sf1 = lookup[i - 1 * exp3_tb[b]];

                    bool c = (sf0.depth == 0) && (sf1.depth == 0) && (sf0.val == sf1.val);
                    uint32_t d = (c ? 0 : (longshot::pow2(num_unfixed) + sf0.depth + sf1.depth));
                    bool v = sf0.val;

                    if (d < min_d)
                    {
                        min_d = d;
                        choice = _dp_item_t(v, d);
                    }

                    x ^= mask;
                }

                lookup[i] = choice;
                rt.next();
            }

            _dp_item_t ans = lookup[exp3_n - 1];

            delete[] lookup;

            return (double)ans.depth / (double)longshot::pow2(num_vars_);
        }
    };

};

#endif // __CIRCUIT_HPP__