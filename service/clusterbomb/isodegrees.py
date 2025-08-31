from ...library.longshot.utils import parse_gate_integer_representation

class FormulaIsodegrees:
    """Extract isomorphism-invariant feature vector from boolean formulas.
    
    This class converts a boolean formula into a canonical feature representation
    that is identical for isomorphic formulas. It counts the occurrences of each
    literal (positive and negative) and creates a sorted tuple that serves as a
    unique signature for the formula's structure, independent of variable naming.
    
    The feature extraction works by:
    1. Counting positive and negative occurrences of each variable
    2. Creating count pairs for each variable: [pos_count, neg_count]
    3. Converting each pair to a sorted tuple: sorted(tuple([pos_count, neg_count]))
    4. Sorting all the sorted tuples to achieve naming-invariance
    5. Converting to immutable tuple for hashing
    
    This ensures that isomorphic formulas (same structure, different variable names)
    produce identical features, making it useful for detecting structural equivalence.
    
    Attributes:
        _posc: List of positive occurrence counts for each variable
        _negc: List of negative occurrence counts for each variable
    
    Example:
        Formula: (x1 ∧ ¬x2) ∨ (x1 ∧ x3) has counts:
        - x1: appears positive 2 times → [2, 0] → sorted tuple (0, 2)
        - x2: appears negative 1 time → [0, 1] → sorted tuple (0, 1)  
        - x3: appears positive 1 time → [1, 0] → sorted tuple (0, 1)
        
        Feature: ((0, 1), (0, 1), (0, 2)) after sorting all sorted tuples
        
        An isomorphic formula (x3 ∧ ¬x1) ∨ (x3 ∧ x2) would produce
        the same feature after sorting.
    """
    def __init__(self, num_vars: int, gates: list[int]):
        """Initialize formula feature extractor.
        
        Args:
            num_vars: Number of variables in the formula
            gates: List of gate representations in integer format
        """
        self._posc = [0] * num_vars
        self._negc = [0] * num_vars
        self.gates = set(gates)
        
        for g in self.gates:
            ld = parse_gate_integer_representation(g).to_dict()
            
            for i in ld['pos']:
                self._posc[i] += 1
            for i in ld['neg']:
                self._negc[i] += 1
    
    def __hash__(self):
        """Return hash of the canonical feature representation."""
        return hash(self.feature)
    
    def __eq__(self, other: 'FormulaIsodegrees'):
        """Check equality between two FormulaFeature instances.
        
        Two FormulaFeature objects are equal if their canonical feature 
        representations are identical, indicating structural equivalence
        of the underlying formulas (isomorphism).
        
        Args:
            other: Another FormulaFeature instance to compare with
            
        Returns:
            bool: True if the formulas are structurally equivalent, False otherwise
        """
        return self.feature == other.feature
    
    def add_gate(self, gate: int) -> None:
        """Add a gate's literal counts to the feature representation.
        
        Parses the gate's integer representation to extract positive and negative
        literals, then increments the corresponding occurrence counts. This method
        allows incremental building of the feature vector by processing gates one
        at a time.
        
        Args:
            gate: Integer representation of a gate/clause containing literals
        """
        if gate in self.gates:
            return
        
        ld = parse_gate_integer_representation(gate).to_dict()
        
        for i in ld['pos']:
            self._posc[i] += 1
        for i in ld['neg']:
            self._negc[i] += 1
            
        self.gates.add(gate)
        
    def remove_gate(self, gate: int) -> None:
        """Remove a gate from the formula feature representation.
        
        Decrements the occurrence counts for all literals present in the gate
        only if the gate exists in the formula. This safely updates both positive
        and negative literal counters based on the gate's integer representation.
        
        Args:
            gate: Integer representation of the gate to remove. The gate is
                parsed to extract its positive and negative literal indices.
        
        Note:
            This method is safe to call even if the gate doesn't exist in the
            formula - it will simply return without modifying any counts.
        """
        if gate not in self.gates:
            return
        
        ld = parse_gate_integer_representation(gate).to_dict()
        
        for i in ld['pos']:
            self._posc[i] -= 1
        for i in ld['neg']:
            self._negc[i] -= 1
        
        self.gates.remove(gate)
        
    @property
    def feature(self):
        """Get the canonical feature tuple."""
        return tuple(
            sorted([
                tuple(sorted(z)) 
                for z in zip(self._posc, self._negc)
            ])
        )
        
    @property
    def num_vars(self):
        """Get the number of variables in the formula"""
        return len(self._posc)