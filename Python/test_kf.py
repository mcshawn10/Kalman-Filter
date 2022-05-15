
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from unittest import TestCase
from kf import KF


class TestKF(TestCase):
    def test_can_construct_with_x_and_v(self):
        x = 0.2
        v = 2.3

        kf = KF(x,v,1.2)
        self.assertAlmostEqual(kf.pos, x)
        self.assertAlmostEqual(kf.vel, v)

    def test_can_predict(self):
        x = 0.2
        v = 2.3

        kf = KF(x,v,1.2)
        kf.predict(dt=0.1)

        self.assertEqual(kf.cov.shape, (2,2))
        self.assertEqual(kf.mean.shape, (2, ))

    def test_can_predict_increases_state_uncertainty(self):
        x = 0.2
        v = 2.3

        kf = KF(x,v,1.2)

        for i in range(10):
            det_before = np.linalg.det(kf.cov)
            kf.predict(dt=0.1)
            det_after = np.linalg.det(kf.cov)

            self.assertGreater(det_after, det_before)

        self.assertEqual(kf.cov.shape, (2,2))
        self.assertEqual(kf.mean.shape, (2, ))

    def test_calling_update_nocrash(self):
        x = 0.2
        v = 2.3

        kf = KF(x,v,1.2)

        kf.update(0.1, 0.1)
