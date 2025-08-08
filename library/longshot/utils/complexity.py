import math

def to_lambda(avgQ: float, *, n: int, eps: float) -> float:
    """
    Convert average query complexity to lambda using a specific formula.
    
    :param avgQ: The average query complexity.
    :param n: The total number of arms.
    :param eps: A small epsilon value to avoid division by zero.
    :return: The lambda value derived from the average query complexity. If avgQ is greater than n, returns None.
    """
    return 1 / (1 - (avgQ - eps) / n) if avgQ <= n else None   
    