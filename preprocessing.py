from __future__ import division
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def overlapSegs(signal, lenSeg, overlapRatio=0):  
  '''Break the 1D signal into overlapping segments in 2D numpy array lenSeg*numSegs'''
  assert(signal.ndim == 1)
  lenSkip = int(lenSeg * (1 - overlapRatio))
  numSegs = (signal.shape[0] - (lenSeg-lenSkip)) // lenSkip 
  
  audioSegs = np.zeros(lenSeg*numSegs).reshape(lenSeg,numSegs)    
  for n in np.arange(numSegs):
    idxs = np.arange(lenSeg)+n*lenSkip
    audioSegs[:, n] = signal[idxs]
  return audioSegs

def powerSpectrum(signal, samplingFreq, fftLength=512, plotEnabled=False):
  '''The signal power spectrum'''
  assert(signal.ndim==1)
  lenAudio = signal.shape[0]
	# Apply Fourier transform
  audioDft = np.fft.fft(signal, fftLength)
  halfLength = np.int(np.ceil((lenAudio + 1) / 2))
  powerSpec = 1/lenAudio * np.square(abs(audioDft[0:halfLength]))

  if lenAudio % 2:
    powerSpec[1:halfLength] *= 2
  else:
    powerSpec[1:halfLength-1] *= 2

  freqs = np.arange(0, halfLength, 1) * (samplingFreq / lenAudio) / 1000.0

  # Plot the signal power spectrum  
  if plotEnabled:    
    plt.figure()
    plt.plot(freqs, powerSpec, color='black')
    plt.xlabel('Freq (kHz)')
    plt.ylabel('Power (linear)')
    plt.draw()

  return powerSpec, freqs

def logPowerSpectrum(signal, samplingFreq, fftLength=512):
  '''The signal log power spectrum'''
  powerSpec, freqs = powerSpectrum(signal, samplingFreq, fftLength)
  powerSpec[powerSpec == 0] = 1e-20
  logPowerSpec = 10 * np.log10(powerSpec)
  return logPowerSpec, freqs


def audioSpectrogram(signal, samplingFreq, fftLength=512, overlapRatio=0.5, plotEnabled=False):
  '''The signal spectrogram of the audio signal'''
  assert(signal.ndim==1)
  audioSegs = overlapSegs(signal, fftLength, overlapRatio)
  numSegs = audioSegs.shape[1]
  audioSpectrogram = np.zeros((fftLength//2 + 1, numSegs))
  for idx in np.arange(numSegs):
    audioSeg = audioSegs[:,idx]
    audioSpectrogram[:,idx], _ = logPowerSpectrum(audioSeg, samplingFreq, False)

  t = np.arange(numSegs) * int(fftLength * (1-overlapRatio)) / samplingFreq
  f = np.arange(0, fftLength/2 + 1) / fftLength * samplingFreq / 1000.0
  tGrid, fGrid = np.meshgrid(t,f)

  if plotEnabled:
    fig = plt.figure()
    ax3 = fig.add_subplot(111)
    ax3.pcolormesh(tGrid,fGrid, 10*np.log10(np.abs(audioSpectrogram)), cmap=cm.jet)
    ax3.axis('tight')
    ax3.set_xlabel('Time(sec)')
    ax3.set_ylabel('Frequency(kHz)')
    plt.show()
    
  return audioSpectrogram, t, f