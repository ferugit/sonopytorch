import numpy as np
import torch

grid_indices = [0, 1, 3, 6, 9, 12, 16, 21, 26, 32, 39, 47, 57, 68, 81, 97, 114, 135, 159, 187, 219, 256]

def chop_array(arr, window_size, hop_size):
    """chop_array([1,2,3], 2, 1) -> [[1,2], [2,3]]"""
    return [arr[i - window_size:i] for i in range(window_size, len(arr) + 1, hop_size)]

def fer_chop_array(arr, window_size, hop_size):
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

def numpy_linspace(left_bound, rigth_bound, steps):
    if steps <= 1:
        return torch.linspace(left_bound, rigth_bound, steps)
    else:
        return torch.linspace(left_bound, rigth_bound, steps + 1)[:-1]


chopped_o = chop_array(grid_indices, 3, 1)
chopped = fer_chop_array(torch.tensor(grid_indices), 3, 1)
assert np.allclose(chopped_o, chopped.data.numpy()), "Chopped differs"

print(chopped_o)
print(chopped)

banks_o = np.zeros([20, 257])
for i, (left, middle, right) in enumerate(chop_array(grid_indices, 3, 1)):
    banks_o[i, left:middle] = np.linspace(0., 1., middle - left, False)
    banks_o[i, middle:right] = np.linspace(1., 0., right - middle, False)


banks = torch.zeros([20, 257])
for i, (left, middle, right) in enumerate(fer_chop_array(torch.tensor(grid_indices), 3, 1)):
    left = left.long().item()
    middle = middle.long().item()
    right = right.long().item()
    banks[i, left:middle] = numpy_linspace(0., 1., (middle - left))
    banks[i, middle:right] = numpy_linspace(1., 0., (right - middle))

assert np.allclose(banks_o, banks.data.numpy()), "Banks differs"