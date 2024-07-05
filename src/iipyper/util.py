import time
from contextlib import contextmanager

from .state import _lock

class profile:
    """simple timing profiler as a decorator or context manager"""
    def __init__(self, label=None, print=print, enable=True):
        self.label = label
        self.print=print
        self.enable=enable

    def __enter__(self):
        if self.enable:
           self.t = time.perf_counter_ns()
        return self

    def __exit__(self, typ, val, tb):
        if self.enable:
            dt = (time.perf_counter_ns() - self.t)*1e-9
            self.print(f'{self.label or "profile"}:\t {int(1000*dt)} ms')

    def __call__(self, f):
        if self.label is None:
            self.label = f.__name__
        def g(*a, **kw):
            with self:
                return f(*a, **kw)
        g.__name__ = f.__name__
        return g

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