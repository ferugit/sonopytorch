import torch

import sonopytorch


def main():

    print("SonopyTorch usage sample\n")

    audio = torch.rand(20, 16000).float()

    print("Power Spectrogram")
    spectrogram = sonopytorch.PowerSpectrogram()
    spec = spectrogram(audio).data.numpy()
    print("SonopyTorch output shape: " + str(spec.shape))

    print("\nMel Spectrogram")
    fs=16000
    mel_spectrogram = sonopytorch.MelSpectrogram(fs)
    mels = mel_spectrogram(audio).data.numpy()
    print("SonopyTorch output shape: " + str(mels.shape))


if __name__ == '__main__':
    main()
