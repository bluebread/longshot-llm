#ifndef __LONGSHOT_CORE_LITERALS_HPP__
#define __LONGSHOT_CORE_LITERALS_HPP__

namespace longshot
{
    class Literals
    {
    private:
        uint32_t pos_;
        uint32_t neg_;
    public:
        Literals(uint32_t p, uint32_t n) : pos_(p), neg_(n) {}

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

        int width() const
        {
            if (is_constant())
                return 0;
            return __builtin_popcount(pos_) + __builtin_popcount(neg_);
        }
    };
}

#endif