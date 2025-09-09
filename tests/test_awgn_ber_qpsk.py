import numpy as np
from nr_urlcc.simulate import simulate_ofdm_awgn_ber
from nr_urlcc.utils import ber_theory_mqam

def test_qpsk_awgn_within_half_db():
    ebn0_db = np.arange(-2, 9, 2, dtype=float)  # -2..8 dB
    sim = simulate_ofdm_awgn_ber(ebn0_db_list=ebn0_db, M=4, bits_per_trial=200_000, rng_seed=123)
    th = ber_theory_mqam(ebn0_db, M=4)
    # Compare at BER around 1e-3..1e-1 region
    # Convert to "effective" Es/N0 shift: find best offset that minimizes RMSE, assert |offset| <= 0.5 dB
    # Simple approach: compute dB error at each point where both are in (1e-5..1e-1)
    mask = (sim > 1e-5) & (th > 1e-5)
    # avoid log of zero
    err_db = 10*np.log10(sim[mask]/th[mask])
    # average magnitude of error
    assert np.mean(np.abs(err_db)) < 0.5
