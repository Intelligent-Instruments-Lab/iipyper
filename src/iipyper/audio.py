from multiprocessing import Pipe, Process
from threading import Thread, Lock
from queue import Queue

import numpy as np

import sounddevice as sd

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


class AudioProcess:
    def __init__(self, **kw):
        """create a separate Process with a main communication thread, audio thread, and compute thread.
        
        Args:
            device: sounddevice device spec (int, str, or pair of)
            samplerate: sounddevice channels spec (None, int or pair of)
            dtype: numpy dtype for audio samples (sounddevice)
            blocksize: audio blocksize (sounddevice)
            samplerate: samplerate (sounddevice)
            use_input: use audio input (default True)
            use_output: use audio ouptut (default True)
            input_block: optional size to buffer input blocks for step
                if you set this, `step` will get a different size block than
                the audio driver `blocksize`
            buffer_frames: number of calls to `step` to buffer ahead
        """
        self.internal_conn, child_iconn = Pipe()
        self.user_conn, child_uconn = Pipe()

        self.proc = Process(
            target=self._process_run, 
            args=(child_iconn, child_uconn), kwargs=kw, 
            daemon=True)
        self.proc.start()

    def init(self, **kw):
        """optionally override in user code
        kw are those passed to __init__, less the audio stream parameters.

        it's possible to update self.audio_params here before the audio stream starts, or set initial values in self.step_params

        Returns:
            returned value is automatically passed to self.send() 
                NOTE: this means the first value of self.recv() in the parent will
                be None, if init does not return or is not implemented.
        """

    def step(self, audio=None, **kw):
        """override in user code.

        you can call `self.send(obj)` from this method to send data back to the
        parent process.

        Args:
            audio: audio input block [blocksize x channels]
            additional keyword arguments are the latest values of anything 
                which has been passed to __call__.

        Returns:
            [channel x time] buffer of audio samples (any length)
        """
        raise NotImplementedError

    def __call__(self, **kw):
        """asynchronously set values in the audio process from the main process"""
        # TODO detect if called from audio process
        if self.proc.is_alive():
            # TODO buffer if not alive
            self.internal_conn.send(kw)

    def _process_run(self, internal_conn, user_conn, **kw):
        self.internal_conn = internal_conn
        self.user_conn = user_conn
        # storage for values set by `__call__`
        # and passed to `step`
        self.step_params = {}

        # lock around reading/writing step_params
        self.lock = Lock()

        self.device=kw.pop('device', None)
        self.samplerate=kw.pop('samplerate', None)
        self.blocksize=kw.pop('blocksize', None)
        self.dtype=kw.pop('dtype', None)
        self.channels=kw.pop('channels', None)

        self.input_block=kw.pop('input_block', None)

        buffer_frames=kw.pop('buffer_frames', 1)

        self.use_input = kw.pop('use_input', True)
        self.use_output = kw.pop('use_output', True)

        ###
        self.send(self.init(**kw))
        ###

        # communication between compute/audio threads
        self.to_step = Queue(0 if self.use_input else buffer_frames)
        self.from_step = Queue(0 if self.use_input else buffer_frames)

        # current audio frame data
        self.in_frame = None
        self.out_frame = None
        self.j_in = 0
        self.j_out = 0

        # support just input/output streams
        if not self.use_input and not self.use_output:
            raise ValueError
        if not self.use_input:
            stream_cls = sd.OutputStream
            cb = lambda *a: self._audio_callback(None, *a)
        elif not self.use_output:
            stream_cls = sd.InputStream
            cb = lambda i, *a: self._audio_callback(i, None, *a)
        else:
            stream_cls = sd.Stream
            cb = self._audio_callback

        self.stream = stream_cls(
            device=self.device,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype=self.dtype,
            channels=self.channels,
            callback=cb, 
        )

        # run compute thread
        self.step_thread = Thread(target=self._step, daemon=True)
        self.step_thread.start()

        # run audio thread
        self.stream.start()

        # run communication loop
        while True:
            d = self.internal_conn.recv()
            with self.lock:
                self.step_params.update(d)

    def send(self, a):
        self.user_conn.send(a)

    def recv(self):
        return self.user_conn.recv()
    
    def callback(self, f):
        """
        decorator to handle messages on the frontend (alternative to recv).
        EXPERIMENTAL. only use this once per app.
        """
        def run():
            while True:
                msg = self.recv()
                f(msg)
        Thread(target=run, daemon=True).start()
        return f
        
    def _step(self):
        while True:
            frame_in = self.to_step.get()
            # copy argument dict under lock
            with self.lock:
                d = dict(**self.step_params)
            if self.use_input:
                d['audio'] = frame_in
            frame_out = self.step(**d)
            if self.use_output and self.use_input:
                if frame_in.shape[1] != frame_out.shape[1]:
                    raise ValueError("input and output audio block sizes don't match")
            if self.use_output:
                self.from_step.put(frame_out)

    def _audio_callback(self, indata, outdata, fs, t, status):
        outdata[:,:] = 0

        # init in_frame
        if self.use_input and self.input_block and self.in_frame is None:
            _, c = self.stream.channels
            self.in_frame = np.zeros((self.input_block, c))

        samps = len(outdata) if indata is None else len(indata)

        # process inputs
        if self.use_input:
            if not self.input_block:
                # just pass the input audio frame directly
                self.to_step.put(indata)
            else:
                # buffer into length of input_block 
                i_in = 0
                while i_in < samps:
                    samps_to_process = samps-i_in
                    samps_available = self.input_block-self.j_in
                    s = min(samps_to_process, samps_available)
                    i_in += s
                    self.j_in += s

                    self.in_frame[self.j_in-s:self.j_in] = indata[i_in-s:i_in]
                    if self.j_in==len(self.in_frame):
                        self.to_step.put(np.copy(self.in_frame))
                        self.j_in = 0

        # process outputs
        if self.use_output:
            i_out = 0
            while i_out < samps:

                if not self.use_input and not self.to_step.full():
                    # request up to buffer_frames steps
                    self.to_step.put(None)

                if self.out_frame is None:
                    self.out_frame = self.from_step.get()
                    self.j_out = 0

                samps_available = len(self.out_frame)-self.j_out
                samps_to_process = samps-i_out

                s = min(samps_to_process, samps_available)
                i_out += s
                self.j_out += s

                outdata[i_out-s:i_out] = self.out_frame[self.j_out-s:self.j_out]

                if self.j_out==len(self.out_frame):
                    self.out_frame = None


