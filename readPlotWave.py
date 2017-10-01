import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

sampling_freq, audio = wavfile.read('input.wav')

print '\nShape:', audio.shape
print 'Datatype:', audio.dtype
print 'Duration:', round(audio.shape[0] / float(sampling_freq), 3), 'seconds'

# Normalize the audio
audio = audio / (2.**15)

# audio = audio[:30]
x_values = np.arange(0, len(audio), 1) / float(sampling_freq)
x_values = x_values * 1000
plt.plot(x_values, audio, color='black')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Audio signal')
plt.show()