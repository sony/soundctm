import math
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ctm")),
)

from soundctm_pq import (
    Q16_ONE,
    cfg_blend_bfp64_pq,
    gamma_step_params_pq_q16,
    get_sigmas_karras_pq_q16,
)


class TestSonySoundCtmPq(unittest.TestCase):
    def test_karras_rho7_pq_schedule_accuracy(self):
        steps = 17
        sigma_min = 0.002
        sigma_max = 80.0
        rho = 7.0

        sigmas_q16 = get_sigmas_karras_pq_q16(steps, sigma_min=sigma_min, sigma_max=sigma_max)
        self.assertEqual(len(sigmas_q16), steps + 1)
        self.assertEqual(sigmas_q16[-1], 0)

        min_inv_rho = sigma_min ** (1.0 / rho)
        max_inv_rho = sigma_max ** (1.0 / rho)
        for i in range(steps):
            ramp = i / float(steps - 1)
            exact = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
            approx = sigmas_q16[i] / float(Q16_ONE)
            rel_err = abs(approx - exact) / max(1e-6, exact)
            self.assertLess(rel_err, 6e-4)

    def test_gamma_step_params_pq_q16_isqrt(self):
        sigma_next = 12.5
        t_min = 0.002
        gamma = 0.9

        s_q16, std_q16 = gamma_step_params_pq_q16(
            int(round(sigma_next * Q16_ONE)),
            int(round(t_min * Q16_ONE)),
            int(round(gamma * Q16_ONE)),
        )
        exact_s = math.sqrt(1.0 - gamma * gamma) * (sigma_next - t_min) + t_min
        exact_std = math.sqrt(sigma_next * sigma_next - exact_s * exact_s)
        self.assertAlmostEqual(s_q16 / float(Q16_ONE), exact_s, delta=3e-4)
        self.assertAlmostEqual(std_q16 / float(Q16_ONE), exact_std, delta=3e-4)

    def test_cfg_blend_bfp64_pq(self):
        text_lat = [math.sin(0.13 * i) * 1.5 for i in range(128)]
        uncond_lat = [math.cos(0.11 * i) * 1.2 for i in range(128)]
        nu = 3.5

        blended = cfg_blend_bfp64_pq(text_lat, uncond_lat, nu=nu)
        for t, u, b in zip(text_lat, uncond_lat, blended):
            exact = u + nu * (t - u)
            self.assertAlmostEqual(b, exact, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
