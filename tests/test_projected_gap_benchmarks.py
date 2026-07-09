from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "run_projected_gap_benchmarks.py"


class ProjectedGapBenchmarksCliTest(unittest.TestCase):
    def test_metadata_reports_band_pairing_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "projected_gap_benchmarks.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--Nk",
                    "5",
                    "--out",
                    str(output_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
            )

            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertNotIn("classify_object", data["settings"])
            self.assertEqual(data["settings"]["reported_object"], "Omega_nu(k)")

            first_row = data["positive_product"][0]
            self.assertIn("winner_pairing_object", first_row)
            self.assertNotIn("winner_component_gamma", first_row)


if __name__ == "__main__":
    unittest.main()
