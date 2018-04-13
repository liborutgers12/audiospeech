from __future__ import division
import numpy as np

import matplotlib.pyplot as plt

import pyaudio
from scipy.io import wavfile

import preprocessing

def plotWaveform(audio, samplingFreq, ax=plt.figure().add_subplot(111)):
  '''Plot the audio waveform in time domain.'''
  x_values = np.arange(0, audio.shape[0], 1) / samplingFreq
  x_values = x_values * 1000
  ax.plot(x_values, audio, 'k')
  ax.set_xlabel('Time (ms)')
  ax.set_ylabel('Amplitude')
  ax.set_title('Audio signal')
  plt.draw()

def plotWaveforms(audio, samplingFreq, fig=plt.figure()):
  '''Plot the waveforms of  in time domain.'''
  if audio.ndim == 1:
    plotWaveform(audio,samplingFreq, ax=fig.add_subplot(111))
  elif audio.ndim == 2:
    ax = fig.add_subplot(211)
    plotWaveform(audio[:,0],samplingFreq,ax)
    ax = fig.add_subplot(212)
    plotWaveform(audio[:,1],samplingFreq,ax)
  plt.show()

def plotPowerSpectrum(audio, samplingFreq):
  '''Plot the audio power spectrum.'''
  preprocessing.audioPowerSpectrum(audio, samplingFreq, plotEnabled=True)

def plotSpectrogram(audio, samplingFreq):
  '''Plot the audio spectrogram.'''
  preprocessing.audioSpectrogram(audio, samplingFreq, plotEnabled=True)

def playBack(audio, samplingFreq):
  audio = audio.astype(np.float32)
  # instantiate PyAudio (1)
  p = pyaudio.PyAudio()

  # open stream (2)
  stream = p.open(format=p.get_format_from_width(audio.dtype.itemsize),
                channels=audio.ndim,
                rate=samplingFreq,
                output=True)
  # play stream (3)
  stream.write(audio.tostring())
  
  # stop stream (4)
  stream.stop_stream()
  stream.close()

  # close PyAudio (5)
  p.terminate()

def playBackFile(fileName):
  samplingFreq, audio = wavfile.read(fileName, 'rb')
  audio = audio / np.amax(np.abs(audio))
  audio = audio.astype(np.float32)
  print('=====Audio information=====')
  print('Shape:', audio.shape)
  print('Datatype:', audio.dtype)
  print('SamplingFrequency:', samplingFreq)
  print('Duration:', round(audio.shape[0] / samplingFreq, 3), 'seconds')

  playBack(audio, samplingFreq)

  