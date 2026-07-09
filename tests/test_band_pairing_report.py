from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "band_pairing_report.py"


class BandPairingReportCliTest(unittest.TestCase):
    def test_report_focuses_on_band_pairing_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--Nk",
                    "5",
                    "--top-modes",
                    "1",
                    "--points",
                    "0.5:1.5",
                    "--out-dir",
                    tmpdir,
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
            self.assertIn("Band pairing mode report", result.stdout)
            self.assertNotIn("Gamma", result.stdout)
            self.assertNotIn("component", result.stdout.lower())

            out_dir = Path(tmpdir)
            summary_path = out_dir / "summary.json"
            report_path = out_dir / "band_pairing_report.csv"
            self.assertTrue(summary_path.exists())
            self.assertTrue(report_path.exists())
            self.assertFalse((out_dir / "mode_summary.csv").exists())
            self.assertFalse((out_dir / "gamma_component_report.csv").exists())
            self.assertFalse((out_dir / "gamma16_definitions.csv").exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["settings"]["basis"], "single-band Omega_nu(k)")
            self.assertNotIn("gamma_definitions", summary)
            self.assertIn("mode_summary", summary)
            self.assertGreater(len(summary["mode_summary"]), 0)

            with report_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertGreater(len(rows), 0)
            self.assertIn("omega_basis", rows[0])
            self.assertIn("omega_phase", rows[0])
            self.assertIn("mode_channel", rows[0])
            self.assertNotIn("leading_gamma", rows[0])
            self.assertNotIn("gamma", rows[0])


if __name__ == "__main__":
    unittest.main()
