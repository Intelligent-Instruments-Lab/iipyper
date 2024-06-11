from threading import Thread
import time
from numbers import Number

import fire

from .util import *
from .timing import *
from .midi import *
from .osc import *
from .audio import *
from .tui import *
from .state import _lock

_threads = []
def repeat(
        interval:float=None, between_calls:bool=False, 
        lock:bool=True, tick:float=5e-3):
    """
    Decorate a function to be called repeatedly in a loop.
    
    Args:
        interval: time in seconds to repeat at.
            If the decorated function returns a number, 
            use that as the interval until the next call
        between_calls: if True, interval is between call and next call,
            if False, between return and next call
        lock: if True, use the global iipyper lock to make calls thread-safe
        tick: minimum interval to sleep for 
            (will spinlock for the remainder for more precise timing)
            if None, always sleep
    """
    # close the decorator over interval and lock arguments
    def decorator(f):
        def g():
            while True:
                t = time.perf_counter()
                returned_interval = maybe_lock(f, lock)

                if isinstance(returned_interval, Number):
                    wait_interval = returned_interval
                else:
                    wait_interval = interval

                # replace False or None with 0
                wait_interval = wait_interval or 0

                if between_calls:
                    # interval is between calls to the decorated function
                    elapsed = time.perf_counter() - t
                    wait = wait_interval - elapsed
                else:
                    # else interval is from now until next call
                    t = time.perf_counter()
                    wait = wait_interval
                # print(f'{wait=}')
                # tt = time.perf_counter()
                if wait >= 0:
                    if tick is None:
                        time.sleep(wait)
                    else:
                        sleep = wait - tick
                        if sleep > 0:
                            time.sleep(sleep)
                        spin_end = t + wait_interval
                        while time.perf_counter() < spin_end: pass
                    # print(f'waited = {time.perf_counter() - tt}')
                else:
                    print(
                        f'@repeat function "{f.__name__}" is late by {-wait}')

        th = Thread(target=g, daemon=True)
        th.start()
        _threads.append(th)

    return decorator

def thread(f):
    """
    EXPERIMENTAL
    wrap a function to be called in a new thread
    """
    def g(*a, **kw):
        th = Thread(target=f, args=a, kwargs=kw, daemon=True)
        th.start()
        _threads.append(th)
    return g

_cleanup_fns = []
# decorator to make a function run on KeyBoardInterrupt (before exit)
def cleanup(f=None):
    """Decorate a function to be called when the iipyper app exits."""
    def decorator(f):
        _cleanup_fns.append(f)
        return f

    if f is None: # return a decorator
        return decorator
    else: #bare decorator case; return decorated function
        return decorator(f)

# locking decorator
def lock(f):
    """wrap the decorated function with the global iipyper lock"""
    def decorated(*a, **kw):
        with _lock:
            f(*a, **kw)
    return decorated

def start_audio():
    """start all audio streams"""
    for a in Audio.instances:
        # print('????')
        if not a.stream.active:
            a.stream.start()

def run(main=None):
    """call this on your main function to run it as an iipyper app"""
    try:
        if main is not None:
            fire.Fire(main)

        # non-blocking main case:
        start_audio()

        # enter a loop if there is not one in main
        while True:
            time.sleep(3e-2)

    except KeyboardInterrupt:
        # for th in _threads:
            # pass
        for a in Audio.instances:
            a.stream.stop()
            a.stream.close()
        for f in _cleanup_fns:
            f()
        exit(0)