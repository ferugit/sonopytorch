# SonopyTorch

*Torch implementation of Sonopy*

Sonopy is a lightweight Python library used to extract audio features developed by MycroftAI. The original implementation can be found [here](https://github.com/MycroftAI/sonopy). By now, this can be used to extract:

 - Power spectrogram
 - Mel spectrogram
 - MFCC

Read more about the library on https://github.com/MycroftAI/sonopy. This implementation can be used while training a torch model as it works with batches of torch tensors.

## Motivation

Torchaudio feature extraction is limited when it comes to end-device deployment, the STFT operation is hardware-dependant. Thus, Sonopy is a very nice alternative for the end-device development as it has the following implementations:

- [python](https://github.com/MycroftAI/sonopy) (original): useful for Raspberry Pi
- [java](https://github.com/mikex86/SonopyJava): useful for Android


For the usage of the model in the end-device one of this two implementations must be used.

## Usage

```python
import torch
import sonopytorch

fs=16000
audio = torch.rand(20, 16000).float() # simulate a batch of 20 audios of 1s

spectrogram = sonopytorch.PowerSpectrogram()
spec = spectrogram(audio)

mel_spectrogram = sonopytorch.MelSpectrogram(fs)
mels = mel_spectrogram(audio)
```
