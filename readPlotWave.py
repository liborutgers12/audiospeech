import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

samplingFreq, audio = wavfile.read('input.wav')

print '\nShape:', audio.shape
print 'Datatype:', audio.dtype
print 'SamplingFrequency:', samplingFreq
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
#plt.show()

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
overlapRatio = 0.5
numSegs = np.floor((len(audio)-fftLength)/(fftLength*overlapRatio)) + 1
audioSpectrogram = np.zeros((fftLength/2 + 1, numSegs))
for idx in np.arange(0, len(audio) - fftLength, fftLength * overlapRatio):
	audioSeg = audio[idx:idx+fftLength]	
	audioSpectrogram[:,int(idx / (fftLength * overlapRatio))] = audioPowerSpectr(		audioSeg, samplingFreq, False)

from mpl_toolkits.mplot3d import Axes3D	
import matplotlib.cm as cm
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')


t = np.arange(0, len(audio) - fftLength, fftLength * overlapRatio) / samplingFreq
f = np.arange(0, fftLength/2 + 1) / float(fftLength) * samplingFreq / 1000.0
tGrid, fGrid = np.meshgrid(t,f)
ax.plot_surface(tGrid,fGrid, np.abs(audioSpectrogram), cmap=cm.jet, linewidth=0)
ax.view_init(azim=-90, elev=90)

fig = plt.figure(2)
ax2 = fig.add_subplot(111)
ax2.specgram(audio)
ax2.axis('tight')

fig = plt.figure(3)
ax3 = fig.add_subplot(111)
ax3.pcolormesh(tGrid,fGrid, 10*np.log10(np.abs(audioSpectrogram)), cmap=cm.jet)
ax3.axis('tight')
ax3.set_xlabel('Time(sec)')
ax3.set_ylabel('Frequency(kHz)')

plt.show()

	




