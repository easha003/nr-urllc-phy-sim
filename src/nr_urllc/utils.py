# SPDX-License-Identifier: MIT
import numpy as np

def erfc_approx(x):
    """Vectorized complementary error function approximation.
    Max abs error ~1.5e-7 for x>=0 (sufficient for BER curves).
    """
    x = np.asarray(x, dtype=float)
    sign = np.where(x < 0, -1.0, 1.0)
    ax = np.abs(x)
    # Coefficients for erf approximation
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    t = 1.0 / (1.0 + p*ax)
    y = (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t
    erf = sign * (1.0 - y * np.exp(-ax*ax))
    return 1.0 - erf



# ---------------- Mod/Demod (Gray) ----------------

def qpsk_mod(bits: np.ndarray):
    assert bits.size % 2 == 0
    b0 = 1 - 2*bits[0::2]
    b1 = 1 - 2*bits[1::2]
    return ((b0 + 1j*b1)/np.sqrt(2)).astype(np.complex64)

def qpsk_demod(symbols: np.ndarray):
    b0 = (symbols.real < 0).astype(np.uint8)
    b1 = (symbols.imag < 0).astype(np.uint8)
    out = np.empty(symbols.size*2, dtype=np.uint8)
    out[0::2] = b0
    out[1::2] = b1
    return out

def qam16_mod(bits: np.ndarray):
    assert bits.size % 4 == 0
    # Gray mapping on I and Q using levels [-3,-1,1,3] / sqrt(10)
    levels = np.array([-3,-1,1,3], dtype=np.float32)/np.sqrt(10.0)
    # Map pairs of bits to one axis
    def map2(b2, b1):
        # 00->-3, 01->-1, 11->1, 10->3  (Gray)
        return np.where((b2==0)&(b1==0), levels[0],
               np.where((b2==0)&(b1==1), levels[1],
               np.where((b2==1)&(b1==1), levels[2], levels[3])))
    b = bits.astype(np.uint8)
    I = map2(b[0::4], b[1::4])
    Q = map2(b[2::4], b[3::4])
    return (I + 1j*Q).astype(np.complex64)

def qam16_demod(symbols: np.ndarray):
    # Quantize to nearest level
    levels = np.array([-3,-1,1,3], dtype=np.float32)/np.sqrt(10.0)
    # decision boundaries midpoints
    bounds = np.array([-2,-0.0,2], dtype=np.float32)/np.sqrt(10.0)
    def decide_axis(x):
        # return bits (b2,b1) per Gray mapping inverse
        # bins: (-inf,-2]->00, (-2,0]->01, (0,2]->11, (2, inf)->10
        b2 = (x > bounds[1]).astype(np.uint8)  # >0 -> b2=1 else 0
        # Determine region 0..3
        r = np.digitize(x, bounds)  # 0..3
        # map region to (b2,b1)
        # regions: 0->00, 1->01, 2->11, 3->10
        b1 = np.array([0,1,1,0], dtype=np.uint8)[r]
        # b2 already mostly correct except region 3 where b2 should be 1, region 0..1 -> 0, region 2..3 -> 1
        b2 = (r >= 2).astype(np.uint8)
        return b2, b1
    b2I, b1I = decide_axis(symbols.real.astype(np.float32))
    b2Q, b1Q = decide_axis(symbols.imag.astype(np.float32))
    out = np.empty(symbols.size*4, dtype=np.uint8)
    out[0::4] = b2I; out[1::4] = b1I; out[2::4] = b2Q; out[3::4] = b1Q
    return out

def modulate(bits: np.ndarray, M: int):
    if M == 4: return qpsk_mod(bits)
    if M == 16: return qam16_mod(bits)
    raise ValueError("Supported M: 4 (QPSK), 16 (16QAM)")

def demodulate(symbols: np.ndarray, M: int):
    if M == 4: return qpsk_demod(symbols)
    if M == 16: return qam16_demod(symbols)
    raise ValueError("Supported M: 4 (QPSK), 16 (16QAM)")

# ---------------- AWGN calibration ----------------

def add_awgn_time_for_ebn0(x_time: np.ndarray, ebn0_db: float, M: int, nfft: int, rng: np.random.Generator):
    """Add complex AWGN to the **time-domain** OFDM stream such that
    after FFT the **per-subcarrier** Eb/N0 equals the target.

    Derivation (numpy fft conventions: FFT = no 1/N, IFFT = 1/N):
      If time-noise ~ CN(0, sigma_t^2), the FFT noise per subcarrier has variance N * sigma_t^2.
      With constellation normalized to Es=1 and k=log2(M), we have Eb/N0 = (Es/N0)/k.
      To target Eb/N0, set Es/N0 = k * 10^(ebn0_db/10).
      Then choose sigma_t^2 = Es / (N * Es/N0) = 1 / (N * Es/N0).
    """
    k = int(np.log2(M))
    esn0_lin = k * (10.0 ** (ebn0_db / 10.0))
    sigma2_t = 1.0 / (nfft * esn0_lin)
    noise = (rng.standard_normal(x_time.shape) + 1j*rng.standard_normal(x_time.shape)) * np.sqrt(sigma2_t/2.0)
    return x_time + noise

# ---------------- Theory curves ----------------

def ber_theory_mqam(ebn0_db: np.ndarray, M: int):
    if M == 4:
        # QPSK (same as BPSK): Pb = Q(sqrt(2*Eb/N0)) = 0.5*erfc(sqrt(Eb/N0))
        x = 10.0**(ebn0_db/10.0)
        return 0.5*erfc_approx(np.sqrt(x))
    if M in (16,):
        # Approx BER for square M-QAM with Gray coding:
        # Pb ≈ 2*(1-1/sqrt(M)) / log2(M) * Q( sqrt(3*log2(M)/(M-1) * Eb/N0) )
        k = np.log2(M)
        x = 10.0**(ebn0_db/10.0)
        from math import sqrt
        alpha = np.sqrt(3*k/ (M-1) * x)
        # Q(z) ≈ 0.5*erfc(z/sqrt(2))
        return 2*(1-1/np.sqrt(M))/k * 0.5*erfc_approx(alpha/np.sqrt(2))
    raise ValueError("Only M in {4,16} supported for theory here.")

def bit_errors(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a.astype(np.uint8) ^ b.astype(np.uint8)))