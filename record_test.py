import pyaudio
import threading
import numpy as np
import time

class MicrophoneRecorder:
    def __init__(self, chunk_size=1024, format=pyaudio.paInt16, channels=1, rate=44100):
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

def process_audio(recorder):
    while True:
        if recorder.frames:
            data = recorder.get_audio_data()
            # Process the audio data here
            print("Processing audio data:", data)
        time.sleep(0.1)

if __name__ == "__main__":
    recorder = MicrophoneRecorder()
    recorder.start_recording()

    processor_thread = threading.Thread(target=process_audio, args=(recorder,))
    processor_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        recorder.stop_recording()
        processor_thread.join()

    