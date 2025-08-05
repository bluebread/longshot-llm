import base64
import struct

def decode_base64_to_float64(encoded_str: str) -> float:
    """
    Decodes a base64 string back to a float64 value. This is useful for retrieving float values from a base64-encoded format.
    
    :param encoded_str: A base64-encoded string representing a float64 value.
    :return: The decoded float64 value.
    :raises ValueError: If the input string is not a valid base64 string or does not decode to 8 bytes."""
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
    """
    Encodes a float64 value to a base64 string. This is useful for storing float values in a compact format.
    
    :param value: The float64 value to encode.
    :return: A base64-encoded string representing the float64 value.
    :raises ValueError: If the input value is not a float.
    """
    if not isinstance(value, float):
        raise ValueError("Value must be a float")
    
    # Pack the float into bytes
    raw = struct.pack(">d", value)  # Use big-endian float64
    return base64.b64encode(raw).decode('ascii')

class Float64Base64:
    """
    A Pydantic model that validates base64-encoded float64 strings.
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: float, *args, **kwargs) -> str:
        return encode_float64_to_base64(v)