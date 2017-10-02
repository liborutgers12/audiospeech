import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

samplingFreq, audio = wavfile.read('input.wav')

print '\nShape:', audio.shape
print 'Datatype:', audio.dtype
print 'Duration:', round(audio.shape[0] / float(samplingFreq), 3), 'seconds'

# Normalize the audio
audio = audio / (2.**15)

# Plot the audio in time domain
# audio = audio[:30]
x_values = np.arange(0, len(audio), 1) / float(samplingFreq)
x_values = x_values * 1000
plt.plot(x_values, audio, color='black')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Audio signal')
plt.show()

def audioPowerSpectr(audio, samplingFreq, plotEnabled):
	lenAudio = len(audio)
	# Apply Fourier transform
	transformedSignal = np.fft.fft(audio)
	halfLength = np.ceil((lenAudio + 1) / 2.0)
	transformedSignal = abs(transformedSignal[0:halfLength])
	transformedSignal /= float(lenAudio)
	transformedSignal **= 2

	lenTransformed = len(transformedSignal)
	if lenAudio % 2:
		transformedSignal[1:lenTransformed] *= 2
	else:
		transformedSignal[1:lenTransformed] *= 2
	
	# Express the power spectrum in dB
	power = 10 * np.log10(transformedSignal)

	# Plot the audio power spectrum
	if plotEnabled:
		x_values = np.arange(0, halfLength, 1) * (samplingFreq / lenAudio) / 1000.0
		plt.figure()
		plt.plot(x_values, power, color='black')
		plt.xlabel('Freq (in kHz)')
		plt.ylabel('Power (in dB)')
		plt.show()
	return power

# Plot the power spectrum for segments of the audio
fftLength = 512
audioSpectrogram = np.zeros(fftLength, )
for i in np.arange(0, len(audio) - fftLength, fftLength):
	audioSeg = audio[i:i+fftLength]
	audioSpectrogram[:,i] = audioPowerSpectr(audioSeg, samplingFreq, True)
	




