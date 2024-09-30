import pyaudio
import wave
import threading
import numpy as np
import time
from scipy.io.wavfile import write



class AudioPlayer:
    chunk = 1024

    def __init__(self, file):
        """ Init audio stream """ 
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )

    def play(self):
        """ Play entire file """
        data = self.wf.readframes(self.chunk)
        while data != b'':
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)

    def close(self):
        """ Graceful shutdown """ 
        self.stream.close()
        self.p.terminate()


class MicrophoneRecorder:
    def __init__(self, chunk_size=512, format=pyaudio.paInt16, channels=1, rate=16000):
        self.chunk_size = chunk_size
        self.format = format
        self.channels = channels
        self.rate = rate
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        self.frames = []

    def start_recording(self):
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk_size)
        self.recording = True
        self.frames = []

        def record_thread():
            while self.recording:
                data = self.stream.read(self.chunk_size)
                self.frames.append(data)

        self.thread = threading.Thread(target=record_thread)
        self.thread.start()

    def stop_recording(self):
        self.recording = False
        self.thread.join()
        self.stream.stop_stream()
        self.stream.close()

    def get_audio_data(self):
        audio_data = b''.join(self.frames)
        return np.frombuffer(audio_data, dtype=np.int16)
    
    def get_last_audio_chunk(self, size):
        audio_data = b''.join(self.frames[-1:])
        return np.frombuffer(audio_data, dtype=np.int16)
      


def write_wav(filepath:str, data:list, sample_rate:int):    
    write(filepath, sample_rate, data)
