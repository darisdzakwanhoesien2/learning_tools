def store_function(name: str, function_code: str):
    """Store a user-defined function in memory."""
    exec(f"def {name}(x1, x2): return {function_code}", globals())

# Example: Store the default function
store_function("example_function", "2*x1**3 - 6*x1*x2 + 3*x2**2")
