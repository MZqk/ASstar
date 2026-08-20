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
    def test_signed_task_source_names_are_preserved_in_profile_context(self):
        runtime = _Runtime()
        runtime.source_file = Path("/tmp/task/process/working.fit")
        runtime.work_dir = Path("/tmp/task")
        runtime.process_dir = Path("/tmp/task/process")
        runtime.input_dir = Path("/tmp/task/input")
        runtime._task_run_manifest_payload = {
            "source": {
                "selected_path": "/captures/NGC-1579.xisf",
                "files": [
                    {"display_path": "NGC_1579/session1/Light_001.fit"}
                ],
            }
        }

        context = runtime._target_profile_context_text()

        self.assertIn("/captures/NGC-1579.xisf", context)
        self.assertIn("NGC_1579/session1/Light_001.fit", context)

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
        self.assertTrue(
            runtime.pipeline_policy["stage3_background"]["protect_nebulosity"]
        )
        self.assertTrue(
            runtime.pipeline_policy["stage3_background"][
                "reject_samples_on_nebula"
            ]
        )
        self.assertTrue(
            runtime.pipeline_policy["stage4_color"][
                "preserve_emission_context"
            ]
        )
        self.assertTrue(
            runtime.pipeline_policy["stage7_stretch"][
                "star_preserve_with_nebulosity"
            ]
        )
        self.assertTrue(
            runtime.pipeline_policy["secondary_context"][
                "primary_policy_unchanged"
            ]
        )

    def test_frozen_composite_context_survives_metadata_refresh(self):
        runtime = _Runtime()
        initial = {
            "target_name_guess": "Lagoon Nebula",
            "target_type": "bright_emission_reflection_nebula",
            "target_confidence": 0.94,
            "classification_method": "catalog_name_wcs_composite_match",
            "composite_targets": [
                {
                    "name": "Lagoon Nebula",
                    "type": "bright_emission_reflection_nebula",
                },
                {
                    "name": "Trifid Nebula",
                    "type": "bright_emission_reflection_nebula",
                },
            ],
            "secondary_labels": ["large_nebulosity", "emission_red"],
        }
        runtime._sync_runtime_policy_from_profile(initial, source="initial")
        runtime._freeze_primary_target()

        later = {
            "target_name_guess": "Trifid Nebula",
            "target_type": "bright_emission_reflection_nebula",
            "target_confidence": 0.90,
            "classification_method": "catalog_coordinate_match",
            "secondary_labels": ["reflection_blue"],
        }
        runtime._sync_runtime_policy_from_profile(later, source="later")

        self.assertEqual(
            runtime.target_profile["identity_status"],
            "composite_resolved",
        )
        self.assertEqual(
            {item["name"] for item in runtime.target_profile["composite_targets"]},
            {"Lagoon Nebula", "Trifid Nebula"},
        )
        self.assertEqual(
            runtime.pipeline_policy["policy_name"],
            "bright_nebula_hdr_conservative",
        )
        self.assertTrue(
            runtime.target_profile["routing_contract"][
                "composite_target_context_frozen"
            ]
        )


if __name__ == "__main__":
    unittest.main()
