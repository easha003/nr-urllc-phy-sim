# NR‑URLLC PHY Mini Sim (M0–M1)

This repo bootstraps **Milestones M0–M1** of a minimal 5G NR‑URLLC PHY simulator:
- **M0**: scaffold, deterministic utils, folders, interfaces
- **M1**: CP‑OFDM mini‑slot TX/RX on **AWGN**, QPSK/16QAM, **BER vs Eb/N0**

## Quick start (local)

```bash
# (optional) create a venv
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

pip install -r requirements.txt

# Generate BER curves (QPSK & 16QAM) and save plot to artifacts/ber_vs_snr.png
python scripts/run_sims.py

# Run the lightweight unit test (requires pytest)
pytest -q
```

Artifacts are written to `artifacts/` (plots, CSV). Code lives in `src/nr_urlcc/`.

### What’s implemented
- CP‑OFDM framing with configurable `N_fft`, `CP`, `used_subcarriers`, `mini_symbols`
- Centered PRB mapping (no pilots yet)
- QPSK/16QAM Gray mapping, hard demod
- AWGN calibrated to target **Eb/N0** (per subcarrier/symbol) while preserving the OFDM path
- BER simulation & comparison against theory

### Where to go next (M2 preview)
- Add `pilots.py` with LS/MMSE channel estimation and TDL channels in `channel.py`
- Soft demapper (LLRs) for LDPC (M3)
