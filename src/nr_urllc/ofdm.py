# SPDX-License-Identifier: MIT
import numpy as np

def used_indices(nfft: int, used_subcarriers: int):
    assert used_subcarriers <= nfft
    start = (nfft - used_subcarriers) // 2
    return np.arange(start, start + used_subcarriers, dtype=int)

def map_symbols_to_grid(symbols: np.ndarray, nfft: int, used_subcarriers: int, mini_symbols: int):
    """Return resource grid (mini_symbols x nfft) with data placed on centered used subcarriers.
    No pilots yet (M1).
    """
    uidx = used_indices(nfft, used_subcarriers)
    grid = np.zeros((mini_symbols, nfft), dtype=np.complex64)
    per_sym = used_subcarriers
    assert symbols.size == per_sym * mini_symbols, "symbol length mismatch"
    for m in range(mini_symbols):
        grid[m, uidx] = symbols[m * per_sym:(m + 1) * per_sym]
    return grid

def tx_from_grid(grid: np.ndarray, cp: int):
    """IFFT + CP prepend per OFDM symbol; concatenate time stream."""
    mini_symbols, nfft = grid.shape
    out = []
    for m in range(mini_symbols):
        ifft_sym = np.fft.ifft(grid[m])  # numpy ifft includes 1/N scaling
        cp_samples = ifft_sym[-cp:]
        out.append(np.concatenate([cp_samples, ifft_sym]))
    return np.hstack(out).astype(np.complex64)

def rx_to_grid(stream: np.ndarray, nfft: int, cp: int, mini_symbols: int):
    """Slice per-symbol, remove CP, FFT back to grid."""
    grid = np.zeros((mini_symbols, nfft), dtype=np.complex64)
    idx = 0
    for m in range(mini_symbols):
        seg = stream[idx + cp: idx + cp + nfft]
        idx += (nfft + cp)
        grid[m] = np.fft.fft(seg)
    return grid
