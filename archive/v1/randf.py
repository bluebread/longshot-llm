import random
import math
from longshot.literals import NormalFormFormula, FormulaType, Literals

def int_to_bits_shift(n: int, w: int) -> list[int]:
    # extract bit i = (n >> (w-1-i)) & 1 for i=0…w-1
    return [(n >> (w - 1 - i)) & 1 for i in range(w)]

def unrank_combination(n: int, k: int, i: int) -> list[int]:
    """
    Return the i-th (0-based) k-combination of [1..n] in lex order.
    Raises ValueError if i is out of range.
    """
    total = math.comb(n, k)
    if not (0 <= i < total):
        raise ValueError(f"Index i={i} out of range [0, {total})")
    
    result = []
    # We will choose elements one by one.
    # `x` is the next candidate value (starting at 1)
    # `remaining` is how many more elements to pick
    remaining = k
    x = 0
    
    while remaining > 0:
        # How many combinations if we pick x as the next element?
        count_with_x = math.comb(n - x - 1, remaining - 1)
        if i < count_with_x:
            # The i-th combination is among those that **include** x next
            result.append(x)
            remaining -= 1
            # move to the next element after x
            x += 1
        else:
            # skip all combinations starting with x, subtract them from i
            i -= count_with_x
            x += 1
    
    return result

def random_formula(n: int, w: int, s: int, ftype: FormulaType, seed: int | None = None) -> NormalFormFormula:
    """ 
    Generates a random normal form formula (CNF or DNF). 
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(w, int) or w <= 0 or w > n:
        raise ValueError("w must be a positive integer not greater than n")
    if not isinstance(s, int) or s <= 0:
        raise ValueError("s must be a positive integer")
    if ftype not in (FormulaType.Conjunctive, FormulaType.Disjunctive):
        raise ValueError("ftype must be either FormulaType.CNF or FormulaType.DNF")
    if not isinstance(seed, (int, type(None))):
        raise ValueError("seed must be an integer or None")
    
    # Create an empty formula with n variables and type ftype.
    f = NormalFormFormula(n, ftype)
    # Calculate the total possible number of clauses of width w.
    # math.comb(n, w) gives the number of w-sized combinations,
    # then multiplied by 2**w for each binary assignment of signs.
    m =  math.comb(n, w) * (2**w)

    # Initialize random number generator with the given seed, if provided.
    rng = random.Random(seed) if seed is not None else random.Random()
    # Choose `s` unique random numbers between 0 and m-1.
    gates = rng.sample(range(m), s)

    # For each randomly selected gate index, decode its corresponding clause.
    for g in gates:
        # Compute combination index (which combination of settings is chosen)
        gi = g // (2**w)
        # Compute the binary encoding for literal signs within the clause.
        gj = g % (2**w)
        # Retrieve the specific combination (indices of variables) for the clause.
        indices = unrank_combination(n, w, gi)
        # Convert the binary number `gj` into a list of bits for the sign of each literal.
        signs = int_to_bits_shift(gj, w)
        # Separate variables into positive (sign 0) and negative (sign 1) based on bits.
        pos = [v for i, v in enumerate(indices) if signs[i] == 0]
        neg = [v for i, v in enumerate(indices) if signs[i] > 0]
        # Create a Literals object using the positive and negative indices.
        ls = Literals(pos, neg)
        # Toggle the clause in the formula (adds it if not present, removes it if already present).
        f.toggle(ls)
        
    # Return
if __name__ == "__main__":
    # Example usage
    n = 5  # Number of variables
    w = 3  # Width of each clause
    s = 4  # Number of clauses
    ftype = FormulaType.Conjunctive  # Type of formula

    print("n:", n)
    print("w:", w)
    print("s:", s)
    print("ftype:", ftype)
    formula = random_formula(n, w, s, ftype)
    print("Literals:", list(formula.gates))