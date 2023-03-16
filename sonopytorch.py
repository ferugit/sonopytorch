import torch
import math
from functools import lru_cache


def chop_array(arr: torch.Tensor, window_size: int, hop_size: int):
    """chop_array([1,2,3], 2, 1) -> [[1,2], [2,3]]"""
    audio_legth = arr.shape[0]
    chopped_array = torch.empty(
        int(((audio_legth - window_size) / hop_size) + 1), window_size
        )
    idx = 0
    for i in range(window_size, audio_legth + 1, hop_size):
        chopped_array[idx, :] = arr[i - window_size:i]
        idx += 1
    return chopped_array


def numpy_linspace(left_bound: int, rigth_bound: int, steps: int):
    if steps <= 1:
        return torch.linspace(left_bound, rigth_bound, steps)
    else:
        return torch.linspace(left_bound, rigth_bound, steps + 1)[:-1]


def create_dct(n_mfcc: int, n_mels: int, norm: str="ortho") -> torch.Tensor:
    """Create a DCT transformation matrix with shape (``n_mels``, ``n_mfcc``),
    normalized depending on norm.
    .. devices:: CPU
    .. properties:: TorchScript
    Args:
        n_mfcc (int): Number of mfc coefficients to retain
        n_mels (int): Number of mel filterbanks
        norm (str or None): Norm to use (either "ortho" or None)
    Returns:
        Tensor: The transformation matrix, to be right-multiplied to
        row-wise data of size (``n_mels``, ``n_mfcc``).
    """

    if norm is not None and norm != "ortho":
        raise ValueError('norm must be either "ortho" or None')

    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = torch.arange(float(n_mels))
    k = torch.arange(float(n_mfcc)).unsqueeze(1)
    dct = torch.cos(math.pi / float(n_mels) * (n + 0.5) * k)  # size (n_mfcc, n_mels)

    if norm is None:
        dct *= 2.0
    else:
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
    return dct.t()


class MelSpectrogram(torch.nn.Module):

    def __init__(self, sample_rate, window_stride=(160, 80), fft_size=512, num_filt=20, return_spectrogram=False):
        super().__init__()
        self.power_spec = PowerSpectrogram(window_stride=window_stride, fft_size=fft_size)
        self.num_filt = num_filt
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.return_spectrogram = return_spectrogram

    @lru_cache()  # Prevents recalculating when calling with same parameters
    def filterbanks(self, fft_len):
        """Makes a set of triangle filters focused on {num_filter} mel-spaced frequencies"""
        def hertz_to_mels(f):
            return 2595. * torch.log10(torch.tensor(1.) + (f / 700.))

        def mel_to_hertz(mel):
            return 700. * (10**(mel / 2595.) - 1.)

        def correct_grid(x):
            """Push forward duplicate points to prevent useless filters"""
            offset = 0
            for prev, i in zip((x[0] - 1) + x, x):
                offset = max(0, offset + prev + 1 - i)
                yield i + offset

        # Grid contains points for left center and right points of filter triangle
        # mels -> hertz -> fft indices
        grid_mels = torch.linspace(hertz_to_mels(0), hertz_to_mels(self.sample_rate), self.num_filt + 2)
        grid_hertz = mel_to_hertz(grid_mels)
        grid_indices = (grid_hertz * fft_len / self.sample_rate).long()
        grid_indices = list(correct_grid(grid_indices))
        grid_indices = torch.tensor(grid_indices)

        banks = torch.zeros([self.num_filt, fft_len])

        for i, (left, middle, right) in enumerate(chop_array(grid_indices, 3, 1)):
            left = left.long().item()
            middle = middle.long().item()
            right = right.long().item()
            banks[i, left:middle] = numpy_linspace(0., 1., (middle - left))
            banks[i, middle:right] = numpy_linspace(1., 0., (right - middle))
        return banks

    def safe_log(self, x):
        """Prevents error on log(0) or log(-1)"""
        return torch.log(torch.clip(x, torch.finfo(float).eps, None))

    def forward(self, audio):

        # Get batch size
        bs = audio.shape[0]

        # Process Power Spectrogram
        spec = self.power_spec(audio)

        # Initilize result tensor
        mel_spec = torch.empty(bs, spec.shape[1], self.num_filt) # [bs, t, filters]

        # Iterate and get features
        for i in range(bs):
            mel_spec[i, :, :] = self.safe_log(
                torch.matmul(
                    spec[i, :, :],
                    self.filterbanks(spec[i, :, :].shape[1]).T.to(audio.device)
                    )
                )

        if self.return_spectrogram:
            return mel_spec, spec
        
        return mel_spec

class PowerSpectrogram(torch.nn.Module):

    def __init__(self, window_stride=(160, 80), fft_size=512):
        super().__init__()
        self.window_stride = window_stride
        self.fft_size = fft_size

    def forward(self, audio):

        # Get batch size
        bs = audio.shape[0]

        # Initilize result tensor
        time_bins = int(((audio.shape[1] - self.window_stride[0]) / self.window_stride[1]) + 1)
        frequency_bins = int((self.fft_size / 2) + 1)
        spec = torch.empty(bs, time_bins, frequency_bins) # [bs, t, f]

        # Iterate and get features
        for i in range(bs):
            frames = chop_array(audio[i, :], self.window_stride[0], self.window_stride[1])
            fft = torch.fft.rfft(frames, n=self.fft_size)
            spec[i, :, :] = (fft.real**2 + fft.imag**2) / self.fft_size
        
        return spec
        

class MFCC(torch.nn.Module):

    def __init__(self, sample_rate, window_stride=(160, 80),
        fft_size=512, num_filt=20, num_coeffs=13):
        super().__init__()
        self.power_spec = MelSpectrogram(
            sample_rate=sample_rate,
            window_stride=window_stride,
            fft_size=fft_size,
            num_filt=num_filt,
            return_spectrogram=True
            )
        self.dct_mat = create_dct(num_coeffs, num_filt, norm='ortho')

    def safe_log(self, x):
        """Prevents error on log(0) or log(-1)"""
        return torch.log(torch.clip(x, torch.finfo(float).eps, None))

    def forward(self, audio):
        # Get mel-spectrogram
        mels, spec = self.power_spec(audio)
        mfccs = torch.matmul(mels, self.dct_mat.to(audio.device)) # apply the DCT transform
        mfccs[:, :, 0] = self.safe_log(torch.sum(spec, 2))  # Replace first band with log energies
        return mfccs