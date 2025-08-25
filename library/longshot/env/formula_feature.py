from ..utils import parse_gate_integer_representation

class FormulaFeature:
    """Extract isomorphism-invariant feature vector from boolean formulas.
    
    This class converts a boolean formula into a canonical feature representation
    that is identical for isomorphic formulas. It counts the occurrences of each
    literal (positive and negative) and creates a sorted tuple that serves as a
    unique signature for the formula's structure, independent of variable naming.
    
    The feature extraction works by:
    1. Counting positive and negative occurrences of each variable
    2. Pairing these counts for each variable: (pos_count, neg_count)
    3. Sorting the pairs to achieve naming-invariance
    4. Converting to immutable tuple for hashing
    
    This ensures that isomorphic formulas (same structure, different variable names)
    produce identical features, making it useful for detecting structural equivalence.
    
    Attributes:
        _feature: Sorted tuple of (positive_count, negative_count) pairs,
                 providing a canonical representation invariant to variable renaming.
    
    Example:
        Formula: (x1 ∧ ¬x2) ∨ (x1 ∧ x3) has counts:
        - x1: appears positive 2 times → (2, 0)
        - x2: appears negative 1 time → (0, 1)  
        - x3: appears positive 1 time → (1, 0)
        
        Feature: ((0, 1), (1, 0), (2, 0)) after sorting
        
        An isomorphic formula (x3 ∧ ¬x1) ∨ (x3 ∧ x2) would produce
        the same feature after sorting.
    """
    def __init__(self, n: int, gates: list[int]):
        """Initialize formula feature extractor.
        
        Args:
            n: Number of variables in the formula
            gates: List of gate representations in integer format
        """
        posc = [0 for _ in range(n)]
        negc = [0 for _ in range(n)]
        
        for g in gates:
            ld = parse_gate_integer_representation(g).to_dict()
            
            for i in ld['pos']:
                posc[i] += 1
            for i in ld['neg']:
                negc[i] += 1
            
        self._feature = tuple(sorted([tuple(sorted(z)) for z in zip(posc, negc)]))
    
    def __hash__(self):
        """Return hash of the canonical feature representation."""
        return hash(self._feature)
    
    def __eq__(self, other: 'FormulaFeature'):
        """Check equality between two FormulaFeature instances.
        
        Two FormulaFeature objects are equal if their canonical feature 
        representations are identical, indicating structural equivalence
        of the underlying formulas (isomorphism).
        
        Args:
            other: Another FormulaFeature instance to compare with
            
        Returns:
            bool: True if the formulas are structurally equivalent, False otherwise
        """
        return self._feature == other._feature
    
    @property
    def feature(self):
        """Get the canonical feature tuple."""
        return self._feature