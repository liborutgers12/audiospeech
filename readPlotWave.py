import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

sampling_freq, audio = wavfile.read('input.wav')

print '\nShape:', audio.shape
print 'Datatype:', audio.dtype
print 'Duration:', round(audio.shape[0] / float(sampling_freq), 3), 'seconds'

# Normalize the audio
audio = audio / (2.**15)