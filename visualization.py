from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import preprocessing

def plotWaveform(audio, samplingFreq):
  '''Plot the audio waveform in time domain.'''
  x_values = np.arange(0, len(audio), 1) / samplingFreq
  x_values = x_values * 1000
  plt.plot(x_values, audio, color='black')
  plt.xlabel('Time (ms)')
  plt.ylabel('Amplitude')
  plt.title('Audio signal')
  plt.draw()

def plotPowerSpectrum(audio, samplingFreq):
  '''Plot the audio power spectrum.'''
  preprocessing.audioPowerSpectrum(audio, samplingFreq, plotEnabled=True)

def plotSpectrogram(audio, samplingFreq):
  '''Plot the audio spectrogram.'''
  preprocessing.audioSpectrogram(audio, samplingFreq, plotEnabled=True)