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
    protected:
        int size_ = 0;

    public:
        AC0_Circuit();
        ~AC0_Circuit();
        
        int num_vars() const { return num_vars_; }
        int size() const { return size_; }
        int depth() const { return depth_; }
        
        typedef std::bitset<num_vars_> bits_t;

        virtual bool eval(bits_t x) const = 0;
        virtual double avgQ() const = 0;

        virtual bool is_constant() const
        {
            if (size_ == 0) {
                return true;
            }

            int num_inputs = pow2(num_vars_);

            for (int i = 0; i < num_inputs / 2; i++) {
                bits_t x = bits_t(i);
                if (eval(x) != eval(~x)) {
                    return false;
                }
            }
            return true;
        }
    };

    template<int num_vars_>
    class NormalFormFormula : public AC0_Circuit<num_vars_, 2>
    {
        static_assert(num_vars_ > 0, "Number of variables must be greater than 0");
        static_assert(num_vars_ <= 24, "Number of variables must be less than or equal to 24");
    public:
        typedef uint32_t input_t;

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
        ~NormalFormFormula();
        
        int width() const { return width_; }

        void add_clause(Clause cl) {
            if (cl.variables.count() == 0 && cl.negated_variables.count() == 0) {
                return;
            }

            clauses_.push_back(cl);
            size_ += 1;
            int cl_width = __builtin__popcount(cl.variables | cl.negated_variables);
            width = std::max(width, cl_width);

            input_t mask = cl.variables | cl.negated_variables;
            input_t x = 0;

            for (int i = 0; i < pow2(num_vars_ - cl_width); i++) {
                if (type == Type::Disjunctive) {
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
        union _dpt_item_t
        {
            struct
            {
                bool val : 1; // valid if depth_numerator == 0
                uint32_t depth : 31; // represented by an integer due to D_ave(f) * 2^n
            };

            uint32_t raw;

            _dpt_item_t() : raw(0) {}
            _dpt_item_t(bool v, uint32_t dn) : raw(0)
            {
                this->val = v;
                this->depth = dn;
            }
        };
        
        struct _rstr_t 
        {
        protected:
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

    public:
        double avgQ() const {
            uint64_t exp3_n = pow(3, num_vars_);
            uint64_t exp3_tb[num_vars_];

            exp3_tb[0] = 1;

            for (int i = 1; i < num_vars_; i++) {
                exp3_tb[i] = 3 * exp3_tb[i - 1];
            }

            _dp_item_t * lookup = new _dp_item_t[exp3_n];
            _rstr_t<num_vars_> rt;

            for (uint64_t i = 0; i < exp3_n; i++) {
                if (rt.all_fixed()) {
                    lookup[i] = _dp_item_t(eval(rt.vals().to_ulong()), 0);
                    continue;
                }
                
                uint32_t min_d = std::numeric_limits<uint32_t>::max();
                _dpt_item_t choice;
                uint32_t x = rt.unfixed().to_ulong();

                while (x != 0) {
                    unsigned int mask = x & -x;
                    unsigned int b = __builtin_ctz(mask);

                    _dp_item_t sf0 = lookup[(i - 2 * exp3_tb[b])];
                    _dp_item_t sf1 = lookup[(i - 1 * exp3_tb[b])];
                    
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