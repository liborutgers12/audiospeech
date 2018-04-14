from __future__ import division
from __future__ import print_function
import numpy as np
from scipy import fftpack

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def getWinCoeff(winType, lenW):
  if winType == 'RECT':
    winCoeff = np.ones(lenW)
  elif winType == 'TRIANG':
    winCoeff = 1 - np.abs((np.arange(lenW) - (lenW-1)/2)/((lenW-1)/2))
  elif winType == 'HAMMING':
    winCoeff = 0.54 - 0.46*np.cos(2*np.pi/(lenW-1)*np.arange(lenW))
  elif winType == 'HANN':
    winCoeff = 0.5*(1 - np.cos(2*np.pi/(lenW-1)*np.arange(lenW)))
  else:
    raise ValueError('Window type options: RECT or TRIANG or HANN or HAMMING')

  return winCoeff

def overlapSegs(sig, lenSeg, overlapRatio=0, mode='TRUNC', winType='RECT'):  
  '''Break the 1D signal into overlapping segments in 2D numpy array lenSeg*numSegs'''
  assert(sig.ndim == 1)
  lenSkip = int(lenSeg * (1 - overlapRatio))
  numSegs = (sig.shape[0] - (lenSeg-lenSkip)) // lenSkip 
  
  sigSegs = np.zeros(lenSeg*numSegs).reshape(lenSeg,numSegs)    
  for n in np.arange(numSegs):
    idxs = np.arange(lenSeg)+n*lenSkip
    sigSegs[:, n] = sig[idxs]
  
  #Process for different modes
  if mode == 'TRUNC':
    pass
  elif mode == 'PADDING0':
    sigSegs = np.hstack((sigSegs, np.zeros(lenSeg)))
    numSegs += 1
    idxs = np.arange(lenSkip * (numSegs -1), sig.shape[0])
    sigSegs[0:len(idxs),-1] = sig[idxs];
  else:
    raise ValueError('Mode options: TRUNC or PADDING0')

  #Apply window function
  winCoeff = getWinCoeff(winType, lenSeg)
  sigSegs = sigSegs * winCoeff[:,np.newaxis]
  return sigSegs

def myfft(sig, fftLength=512, sides='twosided'):
  if sides == 'twosided':
    return fftpack.fft(sig,fftLength)
  elif sides == 'onesided':
    return np.fft.rfft(sig.real,fftLength)
  else:
    raise ValueError('Sides options: twosided or onesided')

def powerSpectrum(sig, samplingFreq, fftLength=512, sides='onesided', plotEnabled=False):
  '''The signal power spectrum (in V^2)'''
  assert(sig.ndim==1)
  lenSig = sig.shape[0]
	# Apply Fourier transform
  sigDft = myfft(sig, fftLength, sides)
  lenDft = len(sigDft)
  # for power spectral density in V^2/Hz, different scale is needed
  scale = 1.0  # 1.0/samplingFreq
  powerSpec =  scale * np.conjugate(sigDft) * sigDft

  if sides == 'onesided':
    if fftLength % 2:
      powerSpec *= 2
    else:
      # Last point is unpaired Nyquist freq point, don't double
      powerSpec[1:-1] *= 2

  freqs = np.arange(0, lenDft, 1) * (samplingFreq / fftLength) / 1000.0

  # Plot the signal power spectrum  
  if plotEnabled:    
    plt.figure()
    plt.plot(freqs, powerSpec, color='black')
    plt.xlabel('Freq (kHz)')
    plt.ylabel('Power (linear)')
    plt.draw()

  return powerSpec, freqs

def logPowerSpectrum(sig, samplingFreq, fftLength=512):
  '''The signal log power spectrum'''
  powerSpec, freqs = powerSpectrum(sig, samplingFreq, fftLength)
  powerSpec[powerSpec == 0] = 1e-20
  logPowerSpec = 10 * np.log10(powerSpec)
  return logPowerSpec, freqs


def spectrogram(sig, samplingFreq, fftLength=512, overlapRatio=0.5, winType='HAMMING', plotEnabled=False):
  '''The signal spectrogram of the audio signal'''
  assert(sig.ndim==1)
  sigSegs = overlapSegs(sig, fftLength, overlapRatio, winType=winType)
  numSegs = sigSegs.shape[1]
  sigSpectrogram = np.zeros((fftLength//2 + 1, numSegs))
  for idx in np.arange(numSegs):
    sigSeg = sigSegs[:,idx]
    sigSpectrogram[:,idx], _ = logPowerSpectrum(sigSeg, samplingFreq, fftLength)

  t = np.arange(numSegs) * int(fftLength * (1-overlapRatio)) / samplingFreq
  freqs = np.arange(0, fftLength/2 + 1) / fftLength * samplingFreq / 1000.0
  tGrid, fGrid = np.meshgrid(t,freqs)

  if plotEnabled:
    fig = plt.figure()
    ax3 = fig.add_subplot(111)
    ax3.pcolormesh(tGrid,fGrid, sigSpectrogram, cmap=cm.jet)
    ax3.axis('tight')
    ax3.set_xlabel('Time(sec)')
    ax3.set_ylabel('Frequency(kHz)')
    plt.show()
    
  return sigSpectrogram, tGrid, fGrid