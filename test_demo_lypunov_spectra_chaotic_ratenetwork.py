"""Quick, headless regression checks: python -m unittest -v"""
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import demo_lypunov_spectra_chaotic_ratenetwork as demo


class RateLyapunovSpectraTests(unittest.TestCase):
    def run_demo(self, nLE, dns):
        fig = plt.figure()
        self.addCleanup(plt.close, fig)
        with redirect_stdout(io.StringIO()), patch.object(demo, 'show'):
            result = demo.rate_Lyapunov_spectra(
                N=8, g=4, dt=0.1, tSim=2, nLE=nLE, tONS=0.1,
                seedIC=1, seedNet=2, seedONS=3, dns=dns)
        spectrum_ax = next(ax for ax in fig.axes if ax.get_xlabel() == '$i/N$')
        return result, spectrum_ax

    def test_dns_disabled(self):
        for nLE in (1, 3, 8):
            with self.subTest(nLE=nLE):
                disabled, _ = self.run_demo(nLE, False)
                enabled, _ = self.run_demo(nLE, True)
                self.assertEqual(len(disabled), 5)
                self.assertIsNone(disabled[3])
                self.assertIsNone(disabled[4])
                self.assertEqual(disabled[0].shape, (nLE,))
                self.assertEqual(disabled[1].shape, (21, nLE))
                self.assertEqual(disabled[2].shape, (21,))
                for actual, expected in zip(disabled[:3], enabled[:3]):
                    self.assertTrue(np.isfinite(actual).all())
                    np.testing.assert_array_equal(actual, expected)

    def test_partial_spectrum_uses_network_size(self):
        for dns in (False, True):
            with self.subTest(dns=dns):
                result, ax = self.run_demo(3, dns)
                self.assertEqual(len(ax.lines), 2)
                for line in ax.lines:
                    np.testing.assert_array_equal(line.get_xdata(), [0, 1/8, 2/8])
                np.testing.assert_array_equal(ax.lines[0].get_ydata(), result[0])
                np.testing.assert_array_equal(ax.lines[1].get_ydata(), np.zeros(3))

    def test_full_spectrum_dns_regression(self):
        result, ax = self.run_demo(8, True)
        # Recorded from unmodified main (fb818380b09e), with the seeds above.
        np.testing.assert_allclose(result[0], [
            -0.5619744667695771, -0.3791736387886597,
            -0.4106283222977742, -1.2195172255635536,
            -1.0006460096764025, -1.1580565523883442,
            -1.9708313334602360, -1.4226515919513039,
        ], rtol=1e-9, atol=1e-11)
        # DNS uses finite differences, so allow for subtraction roundoff.
        self.assertAlmostEqual(result[3], -0.20242956316304217, delta=1e-6)
        self.assertEqual(result[4].shape, (21,))
        self.assertTrue(np.isfinite(result[4]).all())
        for line in ax.lines:
            np.testing.assert_array_equal(line.get_xdata(), np.arange(8)/8)


if __name__ == '__main__':
    unittest.main()
