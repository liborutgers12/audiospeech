from __future__ import division 
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import preprocessing
import visualization

samplingFreq, audio = wavfile.read('input.wav')

print(max(audio))
print('Shape:', audio.shape)
print('Datatype:', audio.dtype)
print('SamplingFrequency:', samplingFreq)
print('Duration:', round(audio.shape[0] / samplingFreq, 3), 'seconds')

# Normalize the audio
audio = audio / max(abs(audio))

# Plot the audio in time domain
visualization.plotWaveform(audio, samplingFreq)

fftLength = 512
overlapRatio = 0.5
preprocessing.audioSpectrogram(audio,samplingFreq,fftLength,overlapRatio,True)
