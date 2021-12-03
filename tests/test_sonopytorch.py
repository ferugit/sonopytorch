import sys
sys.path.insert(1, sys.path[0].replace('/tests', ''))

import torch
import sonopy
import sonopytorch

import numpy as np

audio = torch.rand(1, 16000).float()

print("Power Spectrogram")

original = sonopy.power_spec(audio[0].data.numpy())
print("Sonopy output shape: " + str(original.shape))

spectrogram = sonopytorch.PowerSpectrogram()
implemented = spectrogram(audio)[0].data.numpy()
print("SonopyTorch output shape: " + str(implemented.shape))

assert np.allclose(original, implemented), "Power Spectrogram differs"

print("\nMel Spectrogram")

fs=16000

original = sonopy.mel_spec(audio[0].data.numpy(), fs)
print("Sonopy output shape: " + str(original.shape))

mel_spectrogram = sonopytorch.MelSpectrogram(fs)
implemented = mel_spectrogram(audio)[0].data.numpy()
print("SonopyTorch output shape: " + str(implemented.shape))

assert np.allclose(original, implemented, rtol=1e-04, atol=1e-07), "Mel-Spectrogram differs"