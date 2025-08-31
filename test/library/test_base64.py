import pytest
import random
from longshot.utils import encode_float64_to_base64, decode_base64_to_float64

def test_encode_decode_roundtrip():
    """Test that encoding and then decoding a float returns the original value."""
    for _ in range(10):
        original_float = random.uniform(-1e9, 1e9)
        encoded = encode_float64_to_base64(original_float)
        decoded = decode_base64_to_float64(encoded)
        assert decoded == original_float

def test_known_values():
    """Test encoding and decoding with known values."""
    test_cases = {
        0.0: 'AAAAAAAAAAA=',
        1.0: 'P/AAAAAAAAA=',
        -1.0: 'v/AAAAAAAAA=',
        3.1415926535: 'QAkh+1RBF0Q=',
    }
    
    for value, encoded_str in test_cases.items():
        assert encode_float64_to_base64(value) == encoded_str
        assert decode_base64_to_float64(encoded_str) == value

def test_decode_invalid_base64_string():
    """Test decoding an invalid base64 string raises ValueError."""
    with pytest.raises(ValueError, match="Invalid base64 string"):
        decode_base64_to_float64("not-a-base64-string!")

def test_decode_incorrect_length():
    """Test decoding a base64 string of incorrect length raises ValueError."""
    with pytest.raises(ValueError, match="Incorrect length for float64 encoding"):
        # "short" encoded in base64 is 'c2hvcnQ=' which is 5 bytes decoded
        decode_base64_to_float64("c2hvcnQ=")

def test_encode_non_float_value():
    """Test encoding a non-float value raises ValueError."""
    with pytest.raises(ValueError, match="Value must be a float"):
        encode_float64_to_base64("not a float")

if __name__ == "__main__":
    pytest.main(['-v', __file__])