import base64
import struct

def decode_base64_to_float64(encoded_str: str) -> float:
    """Decode a base64 encoded float64 value."""
    try:
        raw = base64.b64decode(encoded_str, validate=True)
    except Exception:
        raise ValueError("Invalid base64 string")
    
    if len(raw) != 8:
        raise ValueError("Incorrect length for float64 encoding (should be 8 bytes)")

    # Unpack bytes to float64
    value = struct.unpack(">d", raw)[0]  # Use big-endian float64
    return value

def encode_float64_to_base64(value: float) -> str:
    """Encode a float64 value to base64."""
    if not isinstance(value, float):
        raise ValueError("Value must be a float")
    
    # Pack the float into bytes
    raw = struct.pack(">d", value)  # Use big-endian float64
    return base64.b64encode(raw).decode('ascii')

class Float64Base64:
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: str, *args, **kwargs) -> float:
        return decode_base64_to_float64(v)