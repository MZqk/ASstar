from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


if "sirilpy.exceptions" not in sys.modules:
    package = types.ModuleType("sirilpy")
    exceptions = types.ModuleType("sirilpy.exceptions")

    class SirilError(Exception):
        pass

    class CommandError(SirilError):
        pass

    class DataError(SirilError):
        pass

    exceptions.SirilError = SirilError
    exceptions.CommandError = CommandError
    exceptions.DataError = DataError
    package.exceptions = exceptions
    sys.modules["sirilpy"] = package
    sys.modules["sirilpy.exceptions"] = exceptions

from target_runtime import TargetRuntimeMixin  # noqa: E402


class _Log:
    def info(self, _message):
        pass

    def warn(self, _message):
        pass


class _Runtime(TargetRuntimeMixin):
    def __init__(self):
        self.log = _Log()
        self.target_profile = {}
        self.pipeline_policy = {}
        self._target_primary_frozen = False
        self._frozen_primary_target = {}


class TargetSemanticsTests(unittest.TestCase):
    def test_frozen_cluster_primary_survives_later_nebula_observation(self):
        runtime = _Runtime()
        initial = {
            "target_type": "open_cluster",
            "target_confidence": 0.82,
            "classification_method": "catalog_coordinate_match",
            "secondary_labels": ["dense_star_field"],
        }
        runtime._sync_runtime_policy_from_profile(
            initial,
            source="initial",
        )
        runtime._freeze_primary_target()

        later = {
            "target_type": "bright_emission_reflection_nebula",
            "target_confidence": 0.96,
            "classification_method": "later_plate_solve",
            "secondary_labels": ["large_nebulosity", "emission_red"],
        }
        runtime._sync_runtime_policy_from_profile(
            later,
            source="later",
        )

        self.assertEqual(runtime._active_target_type(), "open_cluster")
        self.assertEqual(
            runtime.pipeline_policy["policy_name"],
            "open_cluster_color_preserve",
        )
        self.assertEqual(
            runtime.target_profile["observed_primary_target"]["type"],
            "bright_emission_reflection_nebula",
        )
        self.assertIn(
            "large_nebulosity",
            runtime.target_profile["secondary_labels"],
        )
        self.assertTrue(runtime.target_profile["primary_target"]["frozen"])


if __name__ == "__main__":
    unittest.main()
