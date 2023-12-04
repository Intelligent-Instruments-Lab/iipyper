import time
from contextlib import contextmanager

from .state import _lock

@contextmanager
def profile(label, print=print):
    t = time.perf_counter_ns()
    yield None
    dt = (time.perf_counter_ns() - t)*1e-9
    print(f'{label}:\t {int(1000*dt)} ms')

def maybe_lock(f, lock, *a, **kw):
    if lock:
        with _lock:
            return f(*a, **kw)
    else:
        return f(*a, **kw)

class Lag:
    def __init__(self, coef_up, coef_down=None, val=None):
        self.coef_up = coef_up
        self.coef_down = coef_down or coef_up
        self.val = val

    def __call__(self, val):
        if self.val is None:
            self.val = val
        else:
            coef = self.coef_up if val > self.val else self.coef_down
            self.val = self.val*coef + val*(1-coef)
        return self.val
    
    def hpf(self, val):
        return val - self(val)

def pascal_to_path(pascal_str):
    return '/'+pascal_str.replace('_', '/')

def path_to_pascal(path_str):
    return path_str[1:].replace('/', '_')

def pascal_to_camel(pascal_str):
    return pascal_str[0].lower() + pascal_str[1:]