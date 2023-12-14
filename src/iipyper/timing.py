import time
from threading import Timer as _Timer

from .util import maybe_lock
    
class Stopwatch:
    def __init__(self, punch:bool=True):
        self.t = None
        if punch:
            self.punch()

    def punch(self, latency:float=0):
        """punch the clock and return elapsed time since previous punch
        
        Args:
            latency: punch `latency` seconds in the past, 
                unless it would be before the previous punch
        """
        t = time.perf_counter_ns() - latency
        if self.t is None:
            dt_ns = 0
        else:
            t = max(self.t, t)
            dt_ns = t - self.t
        self.t = t
        return dt_ns * 1e-9

    def read(self):
        """just return elapsed time since last punch"""
        if self.t is None:
            return self.punch()
        return (time.perf_counter_ns() - self.t) * 1e-9
    
class Timer:
    """a threading.Timer using the global iipyper lock around the timed function.
    also starts automatically by default.
    """
    def __init__(self, interval, f, lock=True, start=True, **kw):
        self.timer = _Timer(max(0,interval), maybe_lock(f, lock), **kw)
        if start:
            self.start()
    def cancel(self):
        self.timer.cancel()
    def start(self):
        self.timer.start()