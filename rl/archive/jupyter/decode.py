l = [1, 5, 11, 13, 29, 31, 37, 41]
l = [i - 1 for i in l]

def decode(x):
    """
    Decode a list of integers into a list of literals.
    """
    # Initialize the result list
    result = ""
    
    # Iterate through the list of integers
    for i in range(4):
        result += str(x % 3)
        x //= 3
    return result[::-1]

print(l)

for i in l:
    print(decode(i))