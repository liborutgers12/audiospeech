from __future__ import division
import pyaudio
from scipy.io import wavfile
import sys
import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import time

def recordMicrophone(
                      nCh = 2,
                      samplingFreq = 16000,
                      maxSec = 5,
                      dType = pyaudio.paInt16,
                      lenChunk = 1024):
  '''Enhanced audio recorder with realtime waveform plots
    ToDo- remove unused multiple figures
    
    Args:
      nCh (int): number of channel to record
      samplingFreq (int): the sampling frequency
      maxSec (float): maximum recording time in secs
      dType (pyaudio format): pyaudio format
      lenChunk (int): chunk size read by pyaudio stream

    Returns:
      audio (numpy array): array storing the recorded audio
      samplingFreq (int): the sampling frequency
  '''
  def getType(dType):
    if dType == pyaudio.paInt8:
      return np.int8
    elif dType == pyaudio.paInt16:
      return np.int16
    elif dType == pyaudio.paFloat32:
      return np.float32

  # instantiate PyAudio (1)
  p = pyaudio.PyAudio()

  # open stream (2)
  stream = p.open(format=dType,
                  channels=nCh,
                  rate=samplingFreq,
                  input=True,
                  frames_per_buffer=lenChunk)

  print('Recording...')

  fig = plt.figure() 
  ax1 = fig.add_subplot(211)
  ax2 = fig.add_subplot(212)
  ax2.set_xlabel('Time (sec)')
  plt.show(block=False)

  # record the stream (3)
  audio = np.array([]).reshape(0, nCh)
  timesec = np.array([])
  while True:
    try:
      data = stream.read(lenChunk)
      data = np.fromstring(data, dtype=getType(dType))   
      
      data = data.reshape(lenChunk,nCh)             
      audio = np.vstack((audio,data))
      
      if timesec.shape[0] == 0:
        basetime = 0
      else:
        basetime = timesec[-1]
      timesec = np.append(timesec, basetime+(np.arange(data.shape[0])+1)/samplingFreq)

      ax1.plot(timesec, audio[:,0], 'k')
      ax2.plot(timesec, audio[:,1], 'b')
      fig.canvas.draw()
      time.sleep(lenChunk/samplingFreq)

    except KeyboardInterrupt:
      break
    if timesec[-1] >= maxSec:
      break

  plt.close('all')

  # stop stream (4)
  stream.stop_stream()
  stream.close()
  p.terminate()

  wavfile.write('./records.wav',samplingFreq,audio)
  print('Recording terminated. Saving to records.wav.')

  return audio, samplingFreq

if __name__ == '__main__':
  recordMicrophone()
