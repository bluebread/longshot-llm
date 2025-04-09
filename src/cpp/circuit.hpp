#ifndef __CIRCUIT_HPP__
#define __CIRCUIT_HPP__

#include <bitset>
#include <vector>
#include <limits>

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
    };

    template<int num_vars_>
    class NormalFormFormula : public AC0_Circuit<num_vars_, 2>
    {
    public:
        enum class Type
        {
            Conjunctive, // Conjunctive Normal Form
            Disjunctive, // Disjunctive Normal Form
        };

        struct Clause
        {
            bits_t variables;
            bits_t negated_variables;
        };
        
        typedef std::bitset<num_vars_> bits_t;

    protected:
        const Type type_ = Type::Disjunctive;

        int width_ = 0;
        std::vector<Clause> clauses_;
        // std::bitset<longshot::pow2(num_vars_)> truth_table_;

    public:
        NormalFormFormula(Type type = Type::Disjunctive) : type_(type)
        {
            assert(type_ == Type::Conjunctive || type_ == Type::Disjunctive);
        }
        ~NormalFormFormula();
        
        int width() const { return width_; }

        void add_clause(Clause cl) {
            if (cl.variables.count() == 0 && cl.negated_variables.count() == 0) {
                return;
            }

            clauses_.push_back(cl);
            size_ += 1;
            cl_width = (cl.variables | cl.negated_variables).count();
            width = std::max(width, cl_width);
        }

        bool eval(bits_t x) const {
            if (type_ == Type::Conjunctive) {
                for (const auto& cl : clauses_) {
                    if ((cl.variables & x) == 0 && (cl.negated_variables & ~x) == 0) {
                        return false;
                    }
                }
                return true;
            } else { // Disjunctive
                for (const auto& cl : clauses_) {
                    if ((cl.variables & x) == cl.variables && (cl.negated_variables & ~x) == cl.negated_variables) {
                        return true;
                    }
                }
                return false;
            }
        }
    private:
        typedef std::pair<bool, double> _dp_item_t; 

    public:
        double avgQ() const {
            unsigned long long exp3_n = pow(3, num_vars_);
            unsigned long long * exp3_tbl = new unsigned long long[num_vars_];

            for (unsigned int i = 0; i < num_vars_; i++) {
                exp3_tbl[i] = pow(3, i);
            }

            _dp_item_t * lookup = new _dp_item_t[exp3_n];
            _rstr_t<num_vars_> rt;

            for (unsigned long long i = 0; i < exp3_n; i++) {
                if (rt.all_fixed()) {
                    lookup[i] = std::make_tuple(eval(rt.vals()), 0.0);
                    continue;
                }
                
                std::vector<int> unfixed = rt.unfixed_bits();

                double min_d = std::numeric_limits<double>::max();
                _dpt_item_t choice;

                for (const int & b : unfixed) {
                    uint64_t sf0 = (i - 2 * exp3_tbl[b]);
                    uint64_t sf1 = (i - 1 * exp3_tbl[b]);
                    const auto & [v0, d0] = lookup[sf0];
                    const auto & [v1, d1] = lookup[sf1];
                    
                    bool c = (d0 == 0.0) && (d1 == 0.0) && (v0 == v1);
                    double d = (c ? 0 : (1.0 + (d0 + d1) / 2));
                    bool v = v0;

                    if (d < min_d) {
                        min_d = d;
                        choice = std::make_tuple(v, d);
                    }
                }

                lookup[i] = choice;
                rt.next();
            }

            const auto & [value, depth] = lookup[exp3_n - 1];

            delete[] lookup;
            delete[] exp3_tbl;

            return depth;
        }
    };

};

#endif // __CIRCUIT_HPP__