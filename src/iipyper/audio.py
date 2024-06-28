# Audio WIP
import sounddevice as sd
# import numpy as np
# import inspect

def audio(**kw):
    """decorator audio callbacks presenting in and out frames as numpy arrays.
    
    this is the simplified form where the callback only takes two arguments.
    sounddevice parameters are passed to the decorator, e.g.
    ```
    @audio(samplerate=48000)
    def _(indata, outdata):
        outdata[:] = 0
    ```
    """
    def decorator(f):
        def callback(indata, outdata, frames, time, status):
            if status:
                print(f'sounddevice error {status=}')
            f(indata, outdata)
        return Audio(callback=callback, **kw)
    return decorator

class Audio:
    """audio stream class with static list of instances. 
    currently wraps sounddevice.Stream.
    """
    instances = [] # 
    def __init__(self, *a, **kw):
        print(sd.query_devices())
        # self.stream = sd.InputStream(*a, **kw) # TODO
        self.stream = sd.Stream(*a, **kw) # TODO
        Audio.instances.append(self)


from multiprocessing import Pipe, Process
from threading import Thread, Lock
from queue import Queue
# from copy import deepcopy
class AudioProcess:
    def __init__(self, **kw):
        """create a separate Process with a main communication thread, audio thread, and compute thread"""
        self.conn, child_conn = Pipe()

        self.proc = Process(
            target=self._process_run, 
            args=(child_conn,), kwargs=kw, 
            daemon=True)
        self.proc.start()

    def init(self, **kw):
        """optionally override in user code
        kw are those passed to __init__, less the audio stream parameters
        """
    def step(self, **kw):
        """override in user code.
        kw are latest values of anything which has been passed to __call__

        Returns:
            [channel x time] buffer of audio samples (any length)
        """
        raise NotImplementedError

    def __call__(self, **kw):
        """asynchronously set values in the audio process from the main process"""
        # TODO detect if called from audio process
        if self.proc.is_alive():
            # TODO buffer if not alive
            self.conn.send(kw)

    def _process_run(self, conn, **kw):
        # storage for values set by `__call__`
        # and passed to `step`
        self.d = {}

        # lock around reading/writing d
        self.lock = Lock()

        # communication between compute/audio threads
        buffer_frames = kw.pop('buffer_frames', 1)
        self.to_step = Queue(buffer_frames)
        self.from_step = Queue(buffer_frames)

        # current audio frame data
        self.audio_frame = None
        self.j = 0

        self.stream = sd.Stream(
            device=kw.pop('device', None),
            samplerate=kw.pop('samplerate', None),
            blocksize=kw.pop('blocksize', None),
            dtype=kw.pop('dtype', None),
            callback=self._audio_callback
        )

        self.init(**kw)

        # run compute thread
        self.step_thread = Thread(target=self._step, daemon=True)
        self.step_thread.start()

        # run audio thread
        self.stream.start()

        # run communication loop
        while True:
            d = conn.recv()
            with self.lock:
                self.d.update(d)
        
    def _step(self):
        while True:
            # print('_step')
            self.to_step.get()
            with self.lock:
                d = dict(**self.d)
            frame = self.step(**d)
            self.from_step.put(frame)

    def _audio_callback(self, indata, outdata, fs, t, status):
        outdata[:,:] = 0
        
        i = 0
        while i < len(outdata):
            if not self.to_step.full():
                # print('put')
                self.to_step.put(True)
            if self.audio_frame is None:
                if not self.from_step.empty(): 
                    self.audio_frame = self.from_step.get()
                    self.j = 0
                else:
                    # print(f'audio: dropped frame')
                    return
            if self.audio_frame is not None:
                samps_to_write = len(outdata)-i
                samps_available = len(self.audio_frame)-self.j
                # print(f'{(samps_to_write, samps_available)=}')
                s = min(samps_to_write, samps_available)

                outdata[i:i+s] = self.audio_frame[self.j:self.j+s]
                i += s
                self.j += s

                if samps_to_write > samps_available:
                    self.audio_frame = None

