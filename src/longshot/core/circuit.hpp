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
#include "truthtable.hpp"

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

    private:
        class Literals
        {
        private:
            uint32_t pos_;
            uint32_t neg_;
        public:
            Literals(uint32_t p, uint32_t n) : pos_(p), neg_(n) {
                if ((pos_ & neg_) > 0)
                    pos_ = neg_ = std::numeric_limits<uint32_t>::max();
            }

            Literals(const Literals &other) : pos_(other.pos_), neg_(other.neg_) {}
            Literals() : pos_(0), neg_(0) {}

            uint32_t pos() const { return pos_; }
            uint32_t neg() const { return neg_; }

            bool is_empty() const
            {
                return (pos_ | neg_) == 0;
            }
            bool is_contradictory() const
            {
                return (pos_ & neg_) > 0;
            }
            bool is_constant() const 
            {
                return is_empty() || is_contradictory();
            }
        };

    protected:
        const Type type_ = Type::Disjunctive;

        int width_ = 0;
        std::vector<Literals> literals_;
        TruthTable truth_table_;

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
            literals_ = other.literals_;
        }
        NormalFormFormula(NormalFormFormula &&other) : 
            AC0_Circuit(other.num_vars_, other.depth_), type_(other.type_), width_(other.width_), truth_table_(other.truth_table_)
        {
            literals_ = std::move(other.literals_);
        }

        ~NormalFormFormula() {}

        Type ftype() const { return type_; }
        int width() const { return width_; }
        const std::vector<Literals> & literals() const { return literals_; }

        void add_clause(Literals ls)
        {
            uint32_t lsp = ls.pos();
            uint32_t lsn = ls.neg();
            int num_pv = __builtin_popcount(lsp);
            int num_nv = __builtin_popcount(lsn);

            if (num_pv == 0 && num_nv == 0)
            {
                return;
            }
            if ((lsp & lsn) > 0)
            {
                // Disjunctive: Literals with "x and not x" is always false
                // Conjunctive: Literals with "x or not x" is always true
                return;
            }

            literals_.push_back(ls);
            this->size_ += 1;
            int cl_width = __builtin_popcount(lsp | lsn);
            width_ = std::max(width_, cl_width);

            input_t mask = lsp | lsn;
            input_t x = 0;

            // TODO: read/write to truth table in 64-bits chunks 
            for (uint64_t i = 0; i < longshot::pow2(num_vars_ - cl_width); i++)
            {
                if (type_ == Type::Disjunctive)
                {
                    input_t y = (x | lsp) & ~lsn;
                    truth_table_.set(y);
                }
                else
                {
                    input_t y = (x | lsn) & ~lsp;
                    truth_table_.reset(y);
                }

                // magic bit manipulation: enumerate all inputs that satisfies the Literals
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