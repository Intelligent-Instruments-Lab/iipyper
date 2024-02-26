import time
from datetime import datetime
from loguru import logger

class OSCLog:
    def __init__(self, osc, **kwargs) -> None:
        self.osc = osc
        self.verbose = kwargs.get('verbose', False)
        # Whether we are either writing a log file or reading one
        self.recording, self.playing = False, False
        # Whether recording/playback are paused
        self.recording_paused, self.playback_paused = False, False
        # Whether to loop at the end of the file
        self.loop_playback = False
        # When the recording was paused, if it is currently
        self.recording_pause_timestamp = 0
        # When the recording started, to calculate relative timestamps
        self.recording_offset_timestamp = 0
        self.playback_start_time = 0 # System timestamp when playback started
        # Offset needed to get from playback file to current time
        self.playback_timestamp_offset = 0
        self.lastEvent_timestamp = 0 # Timestamp of the last event we received
        self.recording_file = kwargs.get('recording_file', None) # File reference we're recording to
        self.playback_file = kwargs.get('playback_file', None) # File reference we're playing from
        self.playback_thread = None # Thread that runs the playback

    """recording"""

    def record_start(self, filename: str=None):
        """Enable logging to a file at the given location."""
        self.setup_recording_file(filename)
        self.recording_offset_timestamp = self._current_time()
        self.recording_paused = False
        self.recording = True
        if self.verbose:
            print(f"[OSCLog] Recording to {self.recording_file}")

    def setup_recording_file(self, filename: str) -> None:
        if filename is None:
            if self.recording_file is None:
                self.recording_file = f"iipyper_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        else:
            self.recording_file = filename
        self.add_sink(self.recording_file, format='{message}', level='INFO')

    def record_pause(self) -> None:
        """Leave file open but pause the recording"""
        self.recording_pause_timestamp = self._current_time()
        self.recording_paused = True
        if self.verbose:
            print("[OSCLog] Recording paused")

    def record_resume(self) -> None:
        """Resume recording after pause"""
        if not self.recording_paused or not self.recording:
            return
        self.recording_offset_timestamp += self._current_time() - self.recording_pause_timestamp
        self.recording_paused = False
        if self.verbose:
            print("[OSCLog] Recording resumed")

    def record_stop(self) -> None:
        """Disable logging and close any open file"""
        if self.recording:
            # logger.remove()
            self.recording = False
            if self.verbose:
                print("[OSCLog] Recording stopped")

    """playback"""

    def playback_start(self, filename: str, transpose: int, time_stretch: float, loop: bool):
        """Start playing back a log file. Also able to transpose everything up/down"""
        if self.verbose:
            print(f"[OSCLog] Starting playback of {filename}")
        self.playback_start_time = self._current_time()
        self.playback_timestamp_offset = self._playback_get_next_timestamp()
        self.loop_playback = loop
        self.playing = True
        if self.playback_thread is None:
            self.playback_thread = self._playback_run_loop()

    def playback_pause(self) -> None:
        """Leave file open but pause playback"""
        self.playback_paused = True
        if self.verbose:
            print("[OSCLog] Playback paused")

    def playback_resume(self) -> None:
        """Resume playback after pause"""
        self.playback_paused = False
        if self.verbose:
            print("[OSCLog] Playback resumed")

    def playback_stop(self) -> None:
        """Stop playback"""
        if self.playing:
            self.playing = False
            if self.verbose:
                print("[OSCLog] Playback stopped")
    
    """logging"""

    def add_sink(self, sink: str, **kwargs) -> None:
        """Add a sink to the logger"""
        logger.add(sink, **kwargs)
    
    def remove_sink(self, sink: str) -> None:
        """Remove a sink from the logger"""
        logger.remove(sink)
    
    def log(self, message: str, level: str='INFO', **kwargs) -> None:
        """Log a message"""
        if self.recording:
            logger.log(level, message)

    # def log_midi_message(self, message: list[int]) -> None:
    #     """Log a MIDI message"""
    #     pass

    # def log_osc_message(self, path: str, types: str, *args) -> None:
    #     """Log an OSC message"""
    #     pass

    """utils"""

    def type_tag(self, msg: list) -> str:
        """Get the type tag of an OSC message"""
        return ''.join([str(type(arg))[8] for arg in msg])

    """private methods"""

    def _restart_playback(self) -> None:
        """Having already started the playback, restart it"""
        if not self.playing:
            return
        self.playback_file.seek(0, 0)
        self.playback_start_time = self._current_time()
        self.playback_timestamp_offset = self._playback_get_next_timestamp()

    def _current_time(self) -> int:
        """Current wall (system) time in microseconds"""
        return int(time.time() * 1e6)

    def _recording_relative_time(self) -> float:
        """Current time relative to start of recording, in seconds"""
        return (self._current_time() - self.recording_offset_timestamp) / 1e6

    def _playback_get_next_timestamp(self) -> float:
        """Get the next timestamp in the playback file"""
        if not self.playing or self.playback_file.closed or self.playback_file.eof:
            return -1.0
        position = self.playback_file.tell()
        timestamp = float(self.playback_file.readline().split()[0])
        self.playback_file.seek(position)
        return timestamp

    def _playback_run_loop(self) -> None:
        """Function that runs in its own thread to time the playback"""
        pass

    """call"""

    def __call__(self, msg: str, *kwargs) -> None:
        self.log(msg, *kwargs)
