from __future__ import division 
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import preprocessing
import utils
import sys

if len(sys.argv) < 2:
	fileName = '/home/boli/Documents/audiospeech/records.wav'
else:
	fileName = sys.argv[1]
samplingFreq, audio = wavfile.read(fileName)

print(np.amax(audio))
print('Shape:', audio.shape)
print('Datatype:', audio.dtype)
print('SamplingFrequency:', samplingFreq)
print('Duration:', round(audio.shape[0] / samplingFreq, 3), 'seconds')

# Normalize the audio
audio = audio / np.amax(np.abs(audio))

# Plot the audio in time domain
utils.plotWaveforms(audio, samplingFreq)
utils.playBack(audio, samplingFreq)

fftLength = 512
overlapRatio = 0.5
if audio.ndim > 1:
	audio1CH = audio[:,0]
else:
	audio1CH = audio
preprocessing.audioSpectrogram(audio1CH,samplingFreq,fftLength,overlapRatio,True)
