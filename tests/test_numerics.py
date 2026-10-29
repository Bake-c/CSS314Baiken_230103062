"""Regression checks independent of the instructor's deliverable verifier."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dense_kernel import naive_ijk, interchanged_ikj
from scaling_study import kernel, build_roofline


def scalar_stencil(matrix, steps):
    current = matrix.copy()
    for _ in range(steps):
        result = current.copy()
        for i in range(1, len(current) - 1):
            for j in range(1, current.shape[1] - 1):
                result[i, j] = (current[i, j] + current[i - 1, j] + current[i + 1, j]
                                + current[i, j - 1] + current[i, j + 1]) / 5.0
        current = result
    return current


class NumericalTests(unittest.TestCase):
    def test_stencil_across_partition_boundaries(self):
        matrix = np.random.default_rng(17).random((11, 9))
        original = matrix.copy()
        # Odd/even step counts exercise both shared buffers. Rows do not divide evenly.
        for steps in (1, 2, 5):
            expected = scalar_stencil(matrix, steps)
            for workers in (1, 2, 4):
                with self.subTest(steps=steps, workers=workers):
                    np.testing.assert_allclose(kernel(matrix, workers, steps), expected,
                                               atol=1e-12, rtol=0)
                    np.testing.assert_array_equal(matrix, original)

    def test_stencil_more_workers_than_rows(self):
        matrix = np.arange(15, dtype=float).reshape(3, 5) ** 2
        np.testing.assert_allclose(kernel(matrix, 8, 3), scalar_stencil(matrix, 3), atol=1e-12, rtol=0)

    def test_dense_kernels(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=(9, 9)), rng.normal(size=(9, 9))
        for function in (naive_ijk, interchanged_ikj):
            np.testing.assert_allclose(function(a, b), a @ b, atol=1e-12, rtol=0)

    def test_roofline_classification_uses_limits(self):
        memory = build_roofline({'cache': {}}, 0, bandwidth=10.0, peak=100.0)
        compute = build_roofline({'cache': {}}, 0, bandwidth=100.0, peak=1.0)
        self.assertEqual(memory['regime'], 'memory-bound')
        self.assertEqual(compute['regime'], 'compute-bound')
        self.assertAlmostEqual(memory['arithmetic_intensity'], 5 / 48)
        for bandwidth in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                build_roofline({'cache': {}}, 0, bandwidth, 100)


if __name__ == '__main__':
    unittest.main()
