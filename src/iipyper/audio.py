# Audio WIP
import sounddevice as sd
# import numpy as np
# import inspect

def audio(**kw):
    def decorator(f):
        def callback(indata, outdata, frames, time, status):
            if status:
                print(f'sounddevice error {status=}')
            f(indata, outdata)
        return Audio(callback=callback, **kw)
    return decorator

class Audio:
    instances = [] # 
    def __init__(self, *a, **kw):
        print(sd.query_devices())
        # self.stream = sd.InputStream(*a, **kw) # TODO
        self.stream = sd.Stream(*a, **kw) # TODO
        Audio.instances.append(self)

    # def callback(self, f):
    #     """ 
    #         decorate function as a sounddevice callback.
    #         indata, outdata are [frames, channels] numpy arrays 
    #     """
    #     args = inspect.signature(f).parameters
    #     def cb(
    #         indata: np.ndarray, outdata: np.ndarray, 
    #         frames: int, time: sd.CData, status: sd.CallbackFlags
    #         ):

    #         self.
