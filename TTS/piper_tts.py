import wave
from piper.voice import PiperVoice

model = "/path/to/model.onnx"
voice = PiperVoice(model)
text = "This is an example of text to speech"
wav_file = wave.open("output.wav", "w")
audio = voice.synthesize(text, wav_file)
