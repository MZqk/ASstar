from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


RUN_ENV = "STARUN_RUN_SIRIL_1_4_4_GOLDEN"
CLI_ENV = "STARUN_SIRIL_CLI"
REFERENCE_VERSION = "1.4.4"
REFERENCE_PIXEL_SHA256 = (
    "fe58eb12c7b11949fff8f98f2b62f7e94ff7882d089d136787cfd2667f1ccce3"
)


class SirilAsinhRGBBlendGoldenTests(unittest.TestCase):
    def test_explicit_rgbblend_matches_implicit_1_4_4_and_saved_signature(self) -> None:
        if os.getenv(RUN_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
            self.skipTest(f"set {RUN_ENV}=1 to run the installed-Siril golden")
        cli = Path(
            os.getenv(
                CLI_ENV,
                "/Applications/Siril.app/Contents/MacOS/siril-cli",
            )
        ).expanduser()
        self.assertTrue(cli.is_file(), f"Siril CLI does not exist: {cli}")
        version = subprocess.run(
            [str(cli), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=30,
        )
        match = re.search(r"\bsiril\s+(\d+\.\d+\.\d+)\b", version.stdout or "", re.I)
        self.assertIsNotNone(match, f"unable to parse Siril version: {version.stdout}")
        self.assertEqual(
            match.group(1),
            REFERENCE_VERSION,
            "configured CLI is not Siril 1.4.4; manually review and regenerate the golden",
        )

        try:
            from astropy.io import fits
        except ImportError as error:  # pragma: no cover - opt-in environment check
            self.fail(f"astropy is required for the opt-in Siril golden: {error}")

        with tempfile.TemporaryDirectory(prefix="aiseestart-siril-asinh-") as directory:
            root = Path(directory)
            yy, xx = np.mgrid[:24, :32]
            x = xx.astype(np.float32) / 31.0
            y = yy.astype(np.float32) / 23.0
            red = np.clip(0.0002 + 0.985 * x + 0.010 * y, 0.0, 0.9998)
            green = np.clip(0.0001 + 0.090 * x + 0.880 * y, 0.0, 0.9997)
            blue = np.clip(0.0003 + 0.720 * (1.0 - x) + 0.270 * y, 0.0, 0.9996)
            red[2:6, 24:30] = 0.99995
            green[10:16, 2:8] = 0.99992
            blue[17:22, 12:20] = 0.99990
            fits.PrimaryHDU(np.stack([red, green, blue]).astype(np.float32)).writeto(
                root / "strong_rgb.fit"
            )

            scripts = {
                "implicit": "asinh 5.25 0.0012",
                "explicit": "asinh 5.25 0.0012 -clipmode=rgbblend",
            }
            for name, asinh_command in scripts.items():
                script_path = root / f"{name}.ssf"
                script_path.write_text(
                    "\n".join(
                        (
                            "requires 1.4.0",
                            f'cd "{root}"',
                            "load strong_rgb",
                            asinh_command,
                            f"save {name}",
                            "close",
                        )
                    )
                    + "\n",
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    [str(cli), "-d", str(root), "-s", str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=60,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout)

            implicit = np.asarray(fits.getdata(root / "implicit.fit"), dtype=np.float32)
            explicit = np.asarray(fits.getdata(root / "explicit.fit"), dtype=np.float32)
            self.assertLessEqual(float(np.max(np.abs(implicit - explicit))), 1e-6)
            pixel_bytes = np.ascontiguousarray(
                explicit.astype("<f4", copy=False)
            ).tobytes()
            self.assertEqual(hashlib.sha256(pixel_bytes).hexdigest(), REFERENCE_PIXEL_SHA256)


if __name__ == "__main__":
    unittest.main()
