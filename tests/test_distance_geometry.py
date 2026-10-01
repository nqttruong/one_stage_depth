import math
import unittest

from ogcde.geometry.camera import CameraIntrinsics


class DistanceGeometryTests(unittest.TestCase):
    def test_optical_axis_scale(self):
        intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        s = intr.scale_from_pixel(320.0, 240.0)
        self.assertAlmostEqual(s, 1.0, places=6)

    def test_off_axis_closed_form(self):
        intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        s = intr.scale_from_pixel(420.0, 240.0)
        expected = math.sqrt(1.0 + ((420.0 - 320.0) / 500.0) ** 2 + 0.0)
        self.assertAlmostEqual(s, expected, places=6)

    def test_crop_intrinsics_update(self):
        intr = CameraIntrinsics(fx=500.0, fy=600.0, cx=320.0, cy=240.0)
        new = intr.crop(10, 20)
        self.assertEqual((new.cx, new.cy), (310.0, 220.0))
        self.assertEqual((new.fx, new.fy), (500.0, 600.0))

    def test_letterbox_intrinsics_update(self):
        intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        new = intr.letterbox(0.5, (25, 30))
        self.assertAlmostEqual(new.fx, 250.0, places=6)
        self.assertAlmostEqual(new.cx, 185.0, places=6)

    def test_hflip_intrinsics_update(self):
        intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        new = intr.hflip(width=640)
        self.assertAlmostEqual(new.cx, 320.0, places=6)


if __name__ == "__main__":
    unittest.main()
