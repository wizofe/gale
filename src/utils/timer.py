import time
from functools import wraps

def timer(func):
    """
    Decorator that measures the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__!r} executed in {end - start:.4f} seconds")
        return result
    return wrapper

class Timer:
    """
    Context manager for timing code blocks.
    """
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        print(f"Block executed in {self.end - self.start:.4f} seconds")
