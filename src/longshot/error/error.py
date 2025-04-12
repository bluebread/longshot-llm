
class LongshotError(Exception):
    """
    Exception raised for errors specific to the Longshot module.

    Attributes:
        message -- explanation of the error
        code -- optional error code or additional metadata
    """

    def __init__(self, message: str, code: int = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        if self.code is not None:
            return f"[Error {self.code}] {self.message}"
        return self.message


# Example usage:
# # from longshot.error import LongshotError
# # def do_longshot_thing(x):
# #     if x < 0:
# #         raise LongshotError("Negative value not allowed", code=400)
#     # ...
