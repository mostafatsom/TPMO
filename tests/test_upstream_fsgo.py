import unittest

import numpy as np

from tpmo import upstream_fsgo_info


class UpstreamFSGOTests(unittest.TestCase):
    def test_imported_upstream_api(self):
        try:
            info = upstream_fsgo_info()
        except ImportError:
            self.skipTest("upstream FSGO is not installed in this environment")
        self.assertEqual(info["module"], "fsgo")
        self.assertIn("fun", info["signature"])
        self.assertIn("grad", info["signature"])
        self.assertIn("precision_step", info["signature"])


if __name__ == "__main__":
    unittest.main()
