# SPDX-License-Identifier: MIT
import numpy as np
from . import ofdm, utils
import math

def simulate_ofdm_awgn_ber(
    ebn0_db_list,
    M=4,
    nfft=256,
    cp=32,
    used_subcarriers=128,
    mini_symbols=7,
    bits_per_trial=20000,
    rng_seed=1,
):
    rng = np.random.default_rng(rng_seed)
    k = int(np.log2(M))
    per_sym = used_subcarriers
    qam_syms_per_frame = mini_symbols * per_sym
    capacity_bits_per_frame = qam_syms_per_frame * k
    nframes = max(1, math.ceil(bits_per_trial / capacity_bits_per_frame))

    results = []
    for ebn0_db in ebn0_db_list:
        bit_errors = 0
        bit_total = 0
        for _ in range(nframes):
            # fresh bits per frame
            tx_bits = rng.integers(0, 2, size=capacity_bits_per_frame, endpoint=True, dtype=np.uint8)
            tx_syms = utils.modulate(tx_bits, M)
            grid = ofdm.map_symbols_to_grid(tx_syms, nfft=nfft, used_subcarriers=used_subcarriers, mini_symbols=mini_symbols)
            tx_time = ofdm.tx_from_grid(grid, cp=cp)

            noisy = utils.add_awgn_time_for_ebn0(tx_time, ebn0_db, M=M, nfft=nfft, rng=rng)
            rx_grid = ofdm.rx_to_grid(noisy, nfft=nfft, cp=cp, mini_symbols=mini_symbols)
            uidx = ofdm.used_indices(nfft, used_subcarriers)
            rx_syms = rx_grid[:, uidx].reshape(-1)
            rx_bits = utils.demodulate(rx_syms, M)[:capacity_bits_per_frame]

            bit_errors += utils.bit_errors(tx_bits, rx_bits)
            bit_total += capacity_bits_per_frame

        results.append(bit_errors / bit_total)
    return np.array(results, dtype=float)
