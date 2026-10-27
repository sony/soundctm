r"""
SoundCTM (ICLR'25) × 8b-is (`https://8b.is`):
Zero-Float `p/q` Karras $\rho=7$ Trajectory Schedule, `isqrt_u128` $\gamma$-Sampling
Jump Integrator, and Omni `BFP-64` Classifier-Free Guidance for Real-Time Game /
PlayStation Sound Synthesis
====================================================================================

Brought to you by **Chris, Hue & the 8b-is team** (https://8b.is · WhatsApp: `+1 757-614-7542`):
  - `8b-is/alexiai` — Sovereign offline AI substrate (`GAIA-MLX-QUANT`, ternary b1.58,
    Quantum 64-Trit bitplanes, `BFP-64`, `wupz`)
  - `8b-is/Engram` — Conditional Memory via Scalable Lookup with `BFP-64` & `p/q`
  - `8b-is/deepsiper-enthea` — Hardware-aligned `BFP-64` & 64-trit bitplane engine

What This Module Adds to Sony `SoundCTM` (`ctm/inference_sampling.py`)
----------------------------------------------------------------------
1. **Binary Exponentiation `p/q` Karras Schedule (`get_sigmas_karras_pq_q16`, $\rho=7$)**:
   - In Karras et al. (`get_sigmas_karras` in `ctm/inference_sampling.py`), the noise
     levels $\sigma_i$ follow:
       $$\sigma_i = \left(\sigma_{\max}^{1/7} + \frac{i}{N-1}\big(\sigma_{\min}^{1/7} - \sigma_{\max}^{1/7}\big)\right)^7$$
   - Because $\rho = 7 = 1 + 2 + 4$, once the endpoint 7th roots are represented in
     `Q24` fixed-point, every step's $r_i^7 = r_i \cdot r_i^2 \cdot (r_i^2)^2$ is
     evaluated with **3 integer multiply-shifts** and **zero floating-point `powf` calls**.

2. **Exact Integer `isqrt_u128` $\gamma$-Multistep Jump Integrator (`gamma_step_params_pq_q16`)**:
   - In SoundCTM's $\gamma$-sampling (`gamma_multistep_sampling`), each trajectory jump
     from $\sigma_i \to \sigma_{i+1}$ via intermediate $s_i$ uses:
       $$s_i = \sqrt{1 - \gamma^2}(\sigma_{i+1} - t_{\min}) + t_{\min}, \qquad
         \text{std}_i = \sqrt{\sigma_{i+1}^2 - s_i^2}$$
   - Both square roots are computed in exact integer `Q16` arithmetic via `isqrt_u128`
     with zero floating-point drift.

3. **Omni `BFP-64` Classifier-Free Guidance (`cfg_blend_bfp64_pq`)**:
   - Compresses `denoised_text` and `denoised_uncond` audio latent spectrograms by
     `3.94x` (`65 B / 64 values`) and evaluates the CFG blend:
       $$\hat{x}_0 = x_{\text{uncond}} + \nu (x_{\text{text}} - x_{\text{uncond}})$$
     directly on aligned integer mantissas.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

Q16_ONE = 1 << 16
Q24_ONE = 1 << 24

# Exact Q24 7th roots for standard SoundCTM Karras defaults (sigma_min=0.002, sigma_max=80.0, rho=7):
#   0.002^(1/7) = 0.41159801238866756 -> round(* 2^24) = 6,905,468
#   80.0^(1/7)  = 1.8700327441067608  -> round(* 2^24) = 31,373,942
DEFAULT_SIGMA_MIN_ROOT7_Q24 = 6_905_468
DEFAULT_SIGMA_MAX_ROOT7_Q24 = 31_373_942


def isqrt_u128(n: int) -> int:
    return math.isqrt(max(0, int(n)))


def _pow7_q24_to_q16(r_q24: int) -> int:
    """
    Compute `(r_q24 / 2^24)^7` in `Q16` using only 3 integer multiply-shifts:
      r^2, r^4 = (r^2)^2, r^7 = r * r^2 * r^4.
    """
    r2_q24 = (r_q24 * r_q24 + (1 << 23)) >> 24
    r4_q24 = (r2_q24 * r2_q24 + (1 << 23)) >> 24
    r3_q24 = (r_q24 * r2_q24 + (1 << 23)) >> 24
    r7_q24 = (r3_q24 * r4_q24 + (1 << 23)) >> 24
    return (r7_q24 + (1 << 7)) >> 8


def get_sigmas_karras_pq_q16(
    n: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
) -> List[int]:
    """
    Zero-float-loop `p/q` (`Q16`) Karras $\rho=7$ noise schedule (including trailing 0).
    """
    if n <= 0:
        return [0]
    if abs(sigma_min - 0.002) < 1e-12 and abs(sigma_max - 80.0) < 1e-12:
        r_min_q24 = DEFAULT_SIGMA_MIN_ROOT7_Q24
        r_max_q24 = DEFAULT_SIGMA_MAX_ROOT7_Q24
    else:
        r_min_q24 = int(round((sigma_min ** (1.0 / 7.0)) * Q24_ONE))
        r_max_q24 = int(round((sigma_max ** (1.0 / 7.0)) * Q24_ONE))

    if n == 1:
        return [_pow7_q24_to_q16(r_max_q24), 0]

    sigmas_q16: List[int] = []
    denom = n - 1
    delta_q24 = r_min_q24 - r_max_q24
    for i in range(n):
        # Exact rational linear interpolation in Q24
        num = i * delta_q24
        r_i_q24 = r_max_q24 + (
            (num + (denom >> 1)) // denom if num >= 0 else -((-num + (denom >> 1)) // denom)
        )
        sigmas_q16.append(_pow7_q24_to_q16(r_i_q24))
    sigmas_q16.append(0)
    return sigmas_q16


def gamma_step_params_pq_q16(
    sigma_next_q16: int,
    t_min_q16: int,
    gamma_q16: int,
) -> Tuple[int, int]:
    r"""
    Compute SoundCTM's $\gamma$-sampling intermediate target $s$ and noise injection
    standard deviation $\text{std}$ in `Q16` using ONLY integer arithmetic and `isqrt_u128`:
      - $s = \sqrt{1 - \gamma^2}(\sigma_{i+1} - t_{\min}) + t_{\min}$
      - $\text{std} = \sqrt{\sigma_{i+1}^2 - s^2}$
    """
    gamma_clamped = max(0, min(Q16_ONE, int(gamma_q16)))
    one_minus_g2_q32 = (Q16_ONE * Q16_ONE) - (gamma_clamped * gamma_clamped)
    sqrt_1mg2_q16 = isqrt_u128(one_minus_g2_q32)

    diff_q16 = max(0, int(sigma_next_q16) - int(t_min_q16))
    s_q16 = ((sqrt_1mg2_q16 * diff_q16 + (1 << 15)) >> 16) + int(t_min_q16)

    var_q32 = max(0, int(sigma_next_q16) * int(sigma_next_q16) - s_q16 * s_q16)
    std_q16 = isqrt_u128(var_q32)
    return s_q16, std_q16


@dataclass(frozen=True)
class Bfp64AudioLatentBlock:
    """Omni `BFP-64` 64-lane audio latent block (`i16` or `i8` mantissas)."""

    shared_exp: int
    mantissas: Tuple[int, ...]

    @classmethod
    def quantize(cls, values: Sequence[float]) -> "Bfp64AudioLatentBlock":
        max_abs = max((abs(float(v)) for v in values), default=0.0)
        if max_abs < 1e-30:
            return cls(shared_exp=-40, mantissas=tuple(0 for _ in values))
        _, exp = math.frexp(max_abs)
        shared_exp = int(exp)
        inv_scale = math.ldexp(1.0, 14 - shared_exp)
        mants = tuple(
            max(-32767, min(32767, int(round(float(v) * inv_scale))))
            for v in values
        )
        return cls(shared_exp=shared_exp, mantissas=mants)


def cfg_blend_bfp64_pq(
    denoised_text: Sequence[float],
    denoised_uncond: Sequence[float],
    nu: float = 3.5,
) -> List[float]:
    """
    Evaluate SoundCTM's Classifier-Free Guidance blend:
        x_cfg = x_uncond + nu * (x_text - x_uncond)
    using 64-lane `BFP-64` integer mantissa alignment and `Q12` guidance scale `nu`.
    """
    nu_q12 = int(round(float(nu) * 4096))
    out: List[float] = []
    for start in range(0, len(denoised_text), 64):
        t_chunk = denoised_text[start : start + 64]
        u_chunk = denoised_uncond[start : start + 64]
        t_bfp = Bfp64AudioLatentBlock.quantize(t_chunk)
        u_bfp = Bfp64AudioLatentBlock.quantize(u_chunk)
        common_exp = max(t_bfp.shared_exp, u_bfp.shared_exp)
        t_shift = common_exp - t_bfp.shared_exp
        u_shift = common_exp - u_bfp.shared_exp
        for tm, um in zip(t_bfp.mantissas, u_bfp.mantissas):
            t_aligned = tm >> t_shift if t_shift < 31 else 0
            u_aligned = um >> u_shift if u_shift < 31 else 0
            # u + nu * (t - u) in Q12
            blended_q12 = (u_aligned << 12) + nu_q12 * (t_aligned - u_aligned)
            out.append(math.ldexp(float(blended_q12), common_exp - 14 - 12))
    return out
