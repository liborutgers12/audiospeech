from __future__ import division
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def overlapSegs(audio, lenSeg, overlapRatio=0):  
  '''Break the 1D audio into overlapping segments in 2D numpy array lenSeg*numSegs'''
  lenSkip = int(lenSeg * (1 - overlapRatio))
  numSegs = (audio.shape[0] - (lenSeg-lenSkip)) // lenSkip 
  if audio.ndim == 1:
    audioSegs = np.zeros(lenSeg*numSegs).reshape(lenSeg,numSegs)    
    for n in np.arange(numSegs):
      idxs = np.arange(lenSeg)+n*lenSkip
      audioSegs[:, n] = audio[idxs]
  return audioSegs

def audioPowerSpectrum(audio, samplingFreq, plotEnabled=False):
  assert(audio.ndim==1)
  lenAudio = audio.shape[0]
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
    audioDft[1:lenTransformed-1] *= 2

  # Express the power spectrum in dB
  powerSpectrum = 10 * np.log10(audioDft)
  freqs = np.arange(0, halfLength, 1) * (samplingFreq / lenAudio) / 1000.0

  # Plot the audio power spectrum
  if plotEnabled:    
    plt.figure()
    plt.plot(freqs, powerSpectrum, color='black')
    plt.xlabel('Freq (in kHz)')
    plt.ylabel('Power (in dB)')
    plt.draw()
  return powerSpectrum, freqs

def audioSpectrogram(audio, samplingFreq, fftLength=512, overlapRatio=0.5, plotEnabled=False):
  '''The audio spectrogram of the audio'''
  assert(audio.ndim==1)
  audioSegs = overlapSegs(audio, fftLength, overlapRatio)
  numSegs = audioSegs.shape[1]
  audioSpectrogram = np.zeros((fftLength//2 + 1, numSegs))
  for idx in np.arange(numSegs):
    audioSeg = audioSegs[:,idx]
    audioSpectrogram[:,idx], _ = audioPowerSpectrum(audioSeg, samplingFreq, False)

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