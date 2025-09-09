#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import sys
# add src/ to path for local execution
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from nr_urlcc.simulate import simulate_ofdm_awgn_ber
from nr_urlcc.utils import ber_theory_mqam

ART = Path(__file__).resolve().parents[1] / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

ebn0 = np.linspace(-2, 12, 8)
ber_qpsk = simulate_ofdm_awgn_ber(ebn0, M=4, bits_per_trial=200_000, rng_seed=42)
ber_16 = simulate_ofdm_awgn_ber(ebn0, M=16, bits_per_trial=400_000, rng_seed=7)

th_qpsk = ber_theory_mqam(ebn0, M=4)
th_16 = ber_theory_mqam(ebn0, M=16)

# Save CSV
with open(ART / "m1_ber_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["EbN0_dB", "BER_QPSK", "BER16QAM", "Theory_QPSK", "Theory16QAM"])
    for i in range(len(ebn0)):
        w.writerow([float(ebn0[i]), float(ber_qpsk[i]), float(ber_16[i]), float(th_qpsk[i]), float(th_16[i])])

# Plot
plt.figure()
plt.semilogy(ebn0, ber_qpsk, marker="o", linestyle="-", label="QPSK (sim)")
plt.semilogy(ebn0, th_qpsk, linestyle="--", label="QPSK (theory)")
plt.semilogy(ebn0, ber_16, marker="s", linestyle="-", label="16QAM (sim)")
plt.semilogy(ebn0, th_16, linestyle="--", label="16QAM (theory)")
plt.grid(True, which="both")
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("BER")
plt.title("M1: OFDM on AWGN — BER vs Eb/N0")
plt.legend()
plt.tight_layout()
plt.savefig(ART / "ber_vs_snr.png", dpi=160)
print("Saved:", ART / "ber_vs_snr.png", "and", ART / "m1_ber_results.csv")
