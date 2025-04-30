#ifndef __BOOL_HPP__
#define __BOOL_HPP__

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
#include <stdexcept>

#include "utils.hpp"
#include "literals.hpp"
#include "truthtable.hpp"

namespace longshot 
{
    class BaseBooleanFunction
    {
    protected:
        int num_vars_;
        
    public:
        typedef uint32_t input_t;
    
        BaseBooleanFunction(int n) : num_vars_(n)
        {
        }

        ~BaseBooleanFunction() = default;
        
        int num_vars() const
        {
            return num_vars_;
        }
        
        virtual bool eval(input_t x) const = 0;

        virtual void as_cnf() = 0;
        virtual void as_dnf() = 0;

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

                    // Subfunctions
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

    class MonotonicBooleanFunction : public BaseBooleanFunction
    {
    private:
        SimpleTruthTable truth_table_;

    public:
        MonotonicBooleanFunction(int n) : 
            BaseBooleanFunction(n), 
            truth_table_(n)
        {
        }
        MonotonicBooleanFunction(const MonotonicBooleanFunction &other) : 
            BaseBooleanFunction(other.num_vars_), 
            truth_table_(other.truth_table_)
        {
        }
        MonotonicBooleanFunction(MonotonicBooleanFunction &&other) : 
            BaseBooleanFunction(other.num_vars_), 
            truth_table_(std::move(other.truth_table_))
        {
        }

        ~MonotonicBooleanFunction() = default;

        void as_cnf()
        {
            truth_table_.set();
        }
        void as_dnf()
        {
            truth_table_.reset();
        }

        bool eval(input_t x) const override
        {
            return truth_table_[x];
        }

        void add_clause(Literals clause)
        {
            _add_literals_core(clause, true);
        }

        void add_term(Literals term)
        {
            _add_literals_core(term, false);
        }

        void _add_literals_core(Literals ls, bool is_clause)
        {
            if (ls.is_constant())
            {
                // Term: Literals with "x and not x" is always false
                // Clause: Literals with "x or not x" is always true
                return;
            }

            uint32_t lsp = ls.pos();
            uint32_t lsn = ls.neg();
            int cl_width = ls.width();

            input_t mask = lsp | lsn;
            input_t x = 0;

            // TODO: read/write to truth table in 64-bits chunks 
            for (uint64_t i = 0; i < longshot::pow2(num_vars_ - cl_width); i++)
            {
                if (is_clause)
                {
                    input_t y = (x | lsn) & ~lsp;
                    truth_table_.reset(y);
                }
                else
                {
                    input_t y = (x | lsp) & ~lsn;
                    truth_table_.set(y);
                }

                // magic bit manipulation: enumerate all inputs that satisfies the Literals
                input_t z = (mask | x) + 1;
                z = z & -z;
                x = (~(z - 1) & x) | z;
            }
        }
    };

    class CountingBooleanFunction : public BaseBooleanFunction
    {
    private:
        CountingTruthTable truth_table_;
        bool is_true_based_;

    public:
        CountingBooleanFunction(int n) : 
            BaseBooleanFunction(n), 
            truth_table_(n), 
            is_true_based_(false)
        {
        }
        CountingBooleanFunction(const CountingBooleanFunction &other) : 
            BaseBooleanFunction(other.num_vars_), 
            truth_table_(other.truth_table_), 
            is_true_based_(other.is_true_based_)
        {
        }
        CountingBooleanFunction(CountingBooleanFunction &&other) : 
            BaseBooleanFunction(other.num_vars_), 
            truth_table_(std::move(other.truth_table_)), 
            is_true_based_(other.is_true_based_)
        {
        }

        ~CountingBooleanFunction() = default;

        bool eval(input_t x) const override
        {
            if (is_true_based_)
                return !truth_table_[x];
            return truth_table_[x];
        }

        void as_cnf()
        {
            is_true_based_ = true;
            truth_table_.reset();
        }

        void as_dnf()
        {
            is_true_based_ = false;
            truth_table_.reset();
        }

        void add_clause(Literals clause)
        {
            _apply_literals_core(clause, true, true);
        }

        void add_term(Literals term)
        {
            _apply_literals_core(term, false, true);
        }

        void del_clause(Literals clause)
        {
            _apply_literals_core(clause, true, false);
        }

        void del_term(Literals term)
        {
            _apply_literals_core(term, false, false);
        }

        void _apply_literals_core(Literals ls, bool is_clause, bool adding)
        {
            if (ls.is_constant() || (is_clause ^ is_true_based_))
            {
                // Term: Literals with "x and not x" is always false
                // Clause: Literals with "x or not x" is always true
                return;
            }

            uint32_t lsp = ls.pos();
            uint32_t lsn = ls.neg();
            int cl_width = ls.width();

            input_t mask = lsp | lsn;
            input_t x = 0;
            
            // TODO: read/write to truth table in 64-bits chunks 
            for (uint64_t i = 0; i < longshot::pow2(num_vars_ - cl_width); i++)
            {
                input_t y = 0;

                if (is_clause)
                    y = (x | lsn) & ~lsp;
                else
                    y = (x | lsp) & ~lsn;

                if (adding)
                    truth_table_.inc(y);
                else
                    truth_table_.dec(y);

                // magic bit manipulation: enumerate all inputs that satisfies the Literals
                input_t z = (mask | x) + 1;
                z = z & -z;
                x = (~(z - 1) & x) | z;
            }
        }
    };
}

#endif