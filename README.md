# audiospeech
This repository contains several source codes for 
* Visualization of audio signal
* Record and play back of audio signal/files
* Preprocessing of audio/speech signal

## Dependency
The following packages/repositories are required
* pyaudio
* numpy and scipy
* matplotlib

## Quick start
* input.wav: The example wavefile.
* [readPlotWave.py](./readPlotWave.py): Examples show the following
  * Load and plot the audio in time domain.
  * Play back the audio 
  * Compute the audio spectrogram and plot.

## What does the repository do?
### Module ```recordMicrophone.py```: An Enhanced Recorder
```python
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
```
### Module ```utils.py```: Utility tools 
Functions include
* plotWaveforms()
* playBack()
* playBackFile()

### Module ```preprocessing.py```: Preprocessing
Functions include
* overlapSegs()
* audioPowerSpectrum()
* audioSpectrogram()

