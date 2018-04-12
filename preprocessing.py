from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def audioPowerSpectrum(audio, samplingFreq, plotEnabled=False):
	lenAudio = len(audio)
	# Apply Fourier transform
	audioDft = np.fft.fft(audio)
	halfLength = np.int(np.ceil((lenAudio + 1) / 2))
	audioDft = abs(audioDft[0:halfLength])
	audioDft /= float(lenAudio)
	audioDft **= 2

	lenTransformed = len(audioDft)
	if lenAudio % 2:
		audioDft[1:lenTransformed] *= 2
	else:
		audioDft[1:lenTransformed] *= 2

	# Express the power spectrum in dB
	powerSpectrum = 10 * np.log10(audioDft)

	# Plot the audio power spectrum
	if plotEnabled:
		x_values = np.arange(0, halfLength, 1) * (samplingFreq / lenAudio) / 1000.0
		plt.figure()
		plt.plot(x_values, powerSpectrum, color='black')
		plt.xlabel('Freq (in kHz)')
		plt.ylabel('Power (in dB)')
		plt.draw()
	return powerSpectrum

def audioSpectrogram(audio, samplingFreq, fftLength=512, overlapRatio=0.5, plotEnabled=False):
  '''The power spectrum for segments of the audio'''
  numSegs = (len(audio)-fftLength)//np.int(np.floor(fftLength*overlapRatio)) + 1
  audioSpectrogram = np.zeros((fftLength//2 + 1, numSegs))
  for idx in np.arange(0, len(audio) - fftLength, np.int(fftLength * overlapRatio)):
    audioSeg = audio[idx:idx+fftLength]
    audioSpectrogram[:,int(idx / (fftLength * overlapRatio))] = \
        audioPowerSpectrum(audioSeg, samplingFreq, False)

  t = np.arange(0, len(audio) - fftLength, fftLength * overlapRatio) / samplingFreq
  f = np.arange(0, fftLength/2 + 1) / float(fftLength) * samplingFreq / 1000.0
  tGrid, fGrid = np.meshgrid(t,f)

  if plotEnabled:
    fig = plt.figure()
    ax3 = fig.add_subplot(111)
    ax3.pcolormesh(tGrid,fGrid, 10*np.log10(np.abs(audioSpectrogram)), cmap=cm.jet)
    ax3.axis('tight')
    ax3.set_xlabel('Time(sec)')
    ax3.set_ylabel('Frequency(kHz)')
    plt.draw()
    
  return audioSpectrogram, t, f