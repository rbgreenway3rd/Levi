from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import pyaudio
from scipy.io.wavfile import write
import torch
import numpy as np
import wave
import struct

 

model = load_silero_vad(onnx=True)


total_samples = 0


chunk = 512  # Record in chunks of 1024 samples
sample_format = pyaudio.paFloat32  # 16 bits per sample
channels = 1
fs = 16000  # Record at 44100 samples per second
seconds = 3
filename = "output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio



stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames = []  # Initialize array to store frames
silero_data = []

print("Recording...")
# Store data in chunks for 3 seconds

for i in range(0, 100):# int(fs / chunk * seconds)):
    data = stream.read(chunk)

    b = bytearray(data)
    f = np.frombuffer(b, dtype=np.float32)
    print(f'max = {f.max()}    min = {f.min()}     type = {type(f[0])}')
    t = torch.from_numpy(f)

    speech_prob = model(t, fs).item()  
    if speech_prob>0.5:
        print(f"SPEAKING: {speech_prob}")
    else:
        print("silent")
    
    frames.append(data)

print("Stopping...")

# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

print(f"type: {type(frames[0])}")  # type -> bytes
print(f"Range: {min(frames)}-{max(frames)}")
print(f"Length: {len(frames)}")

