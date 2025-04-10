#ifndef __CIRCUIT_HPP__
#define __CIRCUIT_HPP__

#include <cassert>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <algorithm>
#include <bitset>

#include <utils.hpp>

namespace longshot
{
    template<int num_vars_, int depth_>
    class AC0_Circuit
    {
        static_assert(num_vars_ > 0, "AC0_Circuit: Number of variables must be greater than 0");
        static_assert(num_vars_ <= 24, "AC0_Circuit: Number of variables must be less than or equal to 24");
        static_assert(depth_ > 0, "AC0_Circuit: Depth must be greater than 0");
    protected:
        int size_ = 0;

    public:
        typedef uint32_t input_t;

        AC0_Circuit() {}
        ~AC0_Circuit() {}
        
        int num_vars() const { return num_vars_; }
        int size() const { return size_; }
        int depth() const { return depth_; }
        
        virtual bool eval(input_t x) const = 0;
        virtual double avgQ() const = 0;

        virtual bool is_constant() const
        {
            if (size_ == 0) {
                return true;
            }

            bool v0 = eval(0);
            uint64_t num_inputs = pow2(num_vars_);

            for (uint64_t i = 1; i < num_inputs; i++) {
                if (eval(i) != v0) {
                    return false;
                }
            }
            return true;
        }
    };

    template<int num_vars_>
    class NormalFormFormula : public AC0_Circuit<num_vars_, 2>
    {
        using input_t = typename AC0_Circuit<num_vars_, 2>::input_t;
    public:

        enum class Type
        {
            Conjunctive, // Conjunctive Normal Form
            Disjunctive, // Disjunctive Normal Form
        };

        struct Clause
        {
            input_t variables;
            input_t negated_variables;
        };
        
    protected:
        const Type type_ = Type::Disjunctive;

        int width_ = 0;
        std::vector<Clause> clauses_;
        std::bitset<longshot::pow2(num_vars_)> truth_table_;

    public:
        NormalFormFormula(Type type = Type::Disjunctive) : type_(type)
        {
            if (type_ == Type::Disjunctive) {
                truth_table_.reset();
            } else {
                truth_table_.set();
            }
        }

        ~NormalFormFormula() {}
        
        int width() const { return width_; }

        void add_clause(Clause cl) {
            int num_pv = __builtin_popcount(cl.variables);
            int num_nv = __builtin_popcount(cl.negated_variables);

            if (num_pv == 0 && num_nv == 0) {
                return;
            }

            clauses_.push_back(cl);
            this->size_ += 1;
            int cl_width = __builtin_popcount(cl.variables | cl.negated_variables);
            width_ = std::max(width_, cl_width);

            input_t mask = cl.variables | cl.negated_variables;
            input_t x = 0;

            for (uint64_t i = 0; i < pow2(num_vars_ - cl_width); i++) {
                if (type_ == Type::Disjunctive) {
                    input_t y = (x | cl.variables) & ~cl.negated_variables;
                    truth_table_.set(y);
                }
                else {
                    input_t y = (x | cl.negated_variables) & ~cl.variables;
                    truth_table_.reset(y);
                }

                // magic bit manipulation: enumerate all inputs that satisfies the clause
                input_t m = __builtin_ctz((mask | x) + 1);
                x = ((0xffffffff << m) & x) | (1 << m);
            }
        }

        bool eval(input_t x) const {
            return truth_table_[x];
        }
    private:
        union _dp_item_t
        {
            struct
            {
                bool val : 1; // valid if depth_numerator == 0
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
            _rstr_t(const _rstr_t & r) : vals_(r.vals_), unfixed_(r.unfixed_) {}

            const input_t & vals() const { return vals_; }
            const input_t & unfixed() const { return unfixed_; }

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

            std::string to_string() {
                std::string s = "";
                for (int i = 0; i < num_vars_; i++)
                {
                    if (unfixed_ & (1u << i)) {
                        s += "?";
                    } else {
                        s += (vals_ & (1u << i)) ? "1" : "0";
                    }
                }

                std::reverse(s.begin(), s.end());

                return s;
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
        double avgQ() const {
            uint64_t exp3_n = pow(3, num_vars_);
            uint64_t exp3_tb[num_vars_];

            exp3_tb[0] = 1;

            for (int i = 1; i < num_vars_; i++) {
                exp3_tb[i] = 3 * exp3_tb[i - 1];
            }

            _dp_item_t * lookup = new _dp_item_t[exp3_n];
            _rstr_t rt;

            for (uint64_t i = 0; i < exp3_n; i++) {

                if (rt.all_fixed()) {
                    bool v = eval(rt.vals());
                    lookup[i] = _dp_item_t(v, 0);
                    rt.next();
                    continue;
                }
                
                uint32_t min_d = std::numeric_limits<uint32_t>::max();
                _dp_item_t choice;
                uint32_t x = rt.unfixed();
                unsigned int num_unfixed = __builtin_popcount(x);

                while (x != 0) {
                    unsigned int mask = x & -x;
                    unsigned int b = __builtin_ctz(mask);

                    _dp_item_t sf0 = lookup[i - 2 * exp3_tb[b]];
                    _dp_item_t sf1 = lookup[i - 1 * exp3_tb[b]];
                    
                    bool c = (sf0.depth == 0) && (sf1.depth == 0) && (sf0.val == sf1.val);
                    uint32_t d = (c ? 0 : (longshot::pow2(num_unfixed) + sf0.depth + sf1.depth));
                    bool v = sf0.val;

                    if (d < min_d) {
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

            return (double)ans.depth / (double)pow2(num_vars_);
        }
    };

};

#endif // __CIRCUIT_HPP__