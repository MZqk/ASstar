import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

if "sirilpy.exceptions" not in sys.modules:
    fake_sirilpy = types.ModuleType("sirilpy")
    fake_exceptions = types.ModuleType("sirilpy.exceptions")
    fake_exceptions.CommandError = RuntimeError
    fake_exceptions.DataError = RuntimeError
    fake_exceptions.SirilError = RuntimeError
    fake_sirilpy.exceptions = fake_exceptions
    sys.modules.setdefault("sirilpy", fake_sirilpy)
    sys.modules.setdefault("sirilpy.exceptions", fake_exceptions)

import display_rendition
from managed_output import (
    _read_managed_display_png,
    audit_display_visibility,
    export_managed_outputs,
)
from review_bundle import create_image_review_bundle
from ui_preview import write_display_preview


class PipelineImportContractTests(unittest.TestCase):
    def test_ui_preview_imports_as_pipeline_namespace_module(self):
        project_root = PIPELINE_DIR.parent
        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(project_root)!r})\n"
            "from pipeline.ui_preview import write_display_preview\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stderr or completed.stdout,
        )


_REVIEW_STAR_COORDINATES = (
    (8, 11),
    (14, 104),
    (20, 70),
    (32, 116),
    (44, 21),
    (61, 42),
    (72, 99),
    (86, 48),
)


def _review_scene(height: int = 96, width: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    nebula = np.exp(
        -0.5
        * (
            ((xx - width * 0.52) / (width * 0.18)) ** 2
            + ((yy - height * 0.48) / (height * 0.20)) ** 2
        )
    )
    image = np.stack(
        (
            0.02 + 0.36 * nebula,
            0.018 + 0.22 * nebula,
            0.021 + 0.30 * nebula,
        )
    ).astype(np.float32)
    for y, x in _REVIEW_STAR_COORDINATES:
        image[:, y, x] = (0.95, 0.88, 0.82)
    return image


def _review_star_visibility_contract() -> tuple[dict, object]:
    count = len(_REVIEW_STAR_COORDINATES)
    reference = {
        "status": "ok",
        "_source_peak_y": np.asarray(
            [position[0] for position in _REVIEW_STAR_COORDINATES],
            dtype=np.int32,
        ),
        "_source_peak_x": np.asarray(
            [position[1] for position in _REVIEW_STAR_COORDINATES],
            dtype=np.int32,
        ),
        "_weak_flags": np.asarray(
            [True, True, True, True, False, False, False, False],
            dtype=bool,
        ),
        "_reference_local_contrast": np.full(count, 0.50, dtype=np.float32),
        "_stage9_visibility_inner_window_size_px": np.full(
            count,
            3,
            dtype=np.int32,
        ),
        "_stage9_visibility_outer_window_size_px": np.full(
            count,
            7,
            dtype=np.int32,
        ),
    }
    config = types.SimpleNamespace(stage9_psf_min_sample_count=4)
    return reference, config


class LinkedReviewRenditionTests(unittest.TestCase):
    def test_mapping_preserves_rgb_direction_and_input(self):
        source = _review_scene()
        original = source.copy()
        contract = display_rendition.build_linked_review_contract(
            source,
            reason="unit_review_only",
            source_stem="stage7_review_with_stars",
        )

        rendered = display_rendition.apply_linked_review_contract(
            source,
            contract,
        )

        np.testing.assert_array_equal(source, original)
        source_direction = source / np.maximum(
            np.sum(source, axis=0, keepdims=True),
            1e-8,
        )
        rendered_direction = rendered / np.maximum(
            np.sum(rendered, axis=0, keepdims=True),
            1e-8,
        )
        visible = np.sum(rendered, axis=0) > 1e-8
        np.testing.assert_allclose(
            rendered_direction[:, visible],
            source_direction[:, visible],
            atol=2e-6,
        )
        self.assertEqual(contract["luminance"]["black_percentile"], 0.002)
        self.assertEqual(contract["luminance"]["white_percentile"], 0.995)
        self.assertEqual(contract["luminance"]["target_median"], 0.18)
        self.assertEqual(contract["input_exposure_state"], "underexposed")
        self.assertTrue(contract["derivative_pixels_changed"])
        self.assertEqual(
            contract["luminance"]["actual_gamma"],
            contract["luminance"]["gamma"],
        )
        self.assertGreaterEqual(contract["luminance"]["gamma"], 0.20)
        self.assertLessEqual(contract["luminance"]["gamma"], 1.00)

    def test_visible_source_selects_identity_contract(self):
        source = np.clip(_review_scene() * 0.72 + 0.14, 0.0, 1.0)
        star_reference, star_config = _review_star_visibility_contract()
        visibility = audit_display_visibility(
            source,
            target_type="bright_emission_reflection_nebula",
            stars_required=True,
            star_reference=star_reference,
            pixel_coordinate_domain="siril_pixel_buffer_bottom_up",
            star_visibility_config=star_config,
        )
        self.assertTrue(visibility["passed"], visibility)
        self.assertEqual(visibility["exposure_state"], "acceptable")

        contract = display_rendition.build_review_contract(
            source,
            reason="field_rotation_residual_review",
            source_stem="stage10_final",
            input_visibility=visibility,
        )
        rendered = display_rendition.apply_review_contract(source, contract)

        self.assertEqual(contract["mode"], "preserve")
        self.assertEqual(
            contract["name"],
            display_rendition.PRESERVE_CONTRACT_NAME,
        )
        self.assertEqual(contract["input_exposure_state"], "acceptable")
        self.assertFalse(contract["derivative_pixels_changed"])
        np.testing.assert_array_equal(rendered, source)

    def test_acceptable_exposure_uses_identity_when_formal_star_gate_is_pending(self):
        source = np.clip(_review_scene() * 0.72 + 0.14, 0.0, 1.0)
        visibility = audit_display_visibility(
            source,
            target_type="bright_emission_reflection_nebula",
            stars_required=True,
        )

        self.assertEqual(visibility["exposure_state"], "acceptable")
        self.assertFalse(visibility["passed"])
        self.assertIn("star_visibility", visibility["failed_checks"])

        contract = display_rendition.build_review_contract(
            source,
            reason="formal_star_gate_pending",
            source_stem="stage7_review_with_stars",
            input_visibility=visibility,
        )

        self.assertEqual(contract["status"], "ready")
        self.assertEqual(contract["mode"], "preserve")
        self.assertTrue(display_rendition.validate_review_contract(contract))

    def test_underexposed_source_maps_once_to_target_median(self):
        source = _review_scene()
        star_reference, star_config = _review_star_visibility_contract()
        visibility = audit_display_visibility(
            source,
            target_type="bright_emission_reflection_nebula",
            stars_required=True,
            star_reference=star_reference,
            pixel_coordinate_domain="siril_pixel_buffer_bottom_up",
            star_visibility_config=star_config,
        )
        self.assertEqual(visibility["exposure_state"], "underexposed")

        contract = display_rendition.build_review_contract(
            source,
            reason="bright_core_starless_rejected_after_recovery",
            source_stem="stage7_review_with_stars",
            input_visibility=visibility,
        )
        rendered = display_rendition.apply_review_contract(source, contract)
        final_visibility = audit_display_visibility(
            rendered,
            target_type="bright_emission_reflection_nebula",
            stars_required=True,
            star_reference=star_reference,
            pixel_coordinate_domain="siril_pixel_buffer_bottom_up",
            star_visibility_config=star_config,
        )

        self.assertEqual(contract["mode"], "linked_visibility_v2")
        self.assertAlmostEqual(
            final_visibility["metrics"]["luminance_median"],
            0.18,
            delta=0.015,
        )
        self.assertTrue(final_visibility["passed"], final_visibility)

    def test_overbright_source_is_not_stretched_again(self):
        source = np.clip(_review_scene() * 0.15 + 0.62, 0.0, 1.0)
        visibility = audit_display_visibility(source)
        self.assertEqual(visibility["exposure_state"], "overexposed")
        self.assertIn("exposure_upper_bounds", visibility["failed_checks"])

        contract = display_rendition.build_review_contract(
            source,
            reason="field_rotation_residual_review",
            source_stem="stage10_final",
            input_visibility=visibility,
        )
        self.assertEqual(contract["status"], "unavailable")
        self.assertIn("overexposed", contract["error"])

        with tempfile.TemporaryDirectory() as temporary:
            report = export_managed_outputs(
                source,
                work_dir=Path(temporary),
                base_filename="result_review",
                output_format="png",
            )
            self.assertFalse(report["ready"])
            self.assertEqual(
                report["display_visibility"]["input_pixels"]["exposure_state"],
                "overexposed",
            )
            self.assertEqual(
                report["display_visibility"]["transform"]["name"],
                "preserve_unmappable_or_overexposed_source",
            )
            decision = report["display_visibility"]["brightening_decision"]
            self.assertFalse(decision["allowed"])
            self.assertEqual(
                decision["refused_reason"],
                "brightening_requires_underexposed_input",
            )
            self.assertEqual(decision["upper_bounds"]["luminance_p10_max"], 0.35)

    def test_known_bright_gray_review_levels_hit_the_upper_gate(self):
        for median_level in (0.505, 0.532, 0.670, 0.668):
            with self.subTest(median_level=median_level):
                source = np.clip(
                    _review_scene() * 0.05 + median_level,
                    0.0,
                    1.0,
                )
                visibility = audit_display_visibility(source)
                contract = display_rendition.build_review_contract(
                    source,
                    reason="known_bright_gray_regression",
                    source_stem="stage10_final",
                    input_visibility=visibility,
                )

                self.assertEqual(
                    visibility["exposure_state"],
                    "overexposed",
                )
                self.assertFalse(visibility["passed"])
                self.assertEqual(contract["status"], "unavailable")
                self.assertFalse(contract["derivative_pixels_changed"])

    def test_subjectless_source_is_unmappable_and_never_brightened(self):
        source = np.full((3, 96, 128), 0.06, dtype=np.float32)
        visibility = audit_display_visibility(source)
        contract = display_rendition.build_review_contract(
            source,
            reason="subject_visibility_missing",
            source_stem="stage10_final",
            input_visibility=visibility,
        )

        self.assertEqual(visibility["exposure_state"], "unmappable")
        self.assertEqual(contract["status"], "unavailable")
        self.assertEqual(contract["input_exposure_state"], "unmappable")
        self.assertFalse(contract["derivative_pixels_changed"])

    def test_star_only_nebula_source_is_not_mistaken_for_mappable_subject(self):
        source = np.full((3, 96, 128), 0.025, dtype=np.float32)
        for y_pos, x_pos in ((8, 11), (20, 70), (44, 21), (72, 99), (86, 48)):
            source[:, y_pos, x_pos] = (0.70, 0.62, 0.54)
        visibility = audit_display_visibility(
            source,
            target_type="bright_emission_reflection_nebula",
            stars_required=True,
        )
        contract = display_rendition.build_review_contract(
            source,
            reason="extended_subject_missing",
            source_stem="stage10_final",
            input_visibility=visibility,
        )

        self.assertTrue(visibility["metrics"]["scene_content_visible"])
        self.assertFalse(visibility["metrics"]["required_subject_mappable"])
        self.assertEqual(visibility["exposure_state"], "unmappable")
        self.assertEqual(contract["status"], "unavailable")

    def test_serialized_contract_replays_exactly(self):
        source = _review_scene()
        contract = display_rendition.build_linked_review_contract(
            source,
            reason="unit_review_only",
            source_stem="stage7_review_with_stars",
        )
        replayed_contract = json.loads(json.dumps(contract))

        first = display_rendition.apply_linked_review_contract(source, contract)
        second = display_rendition.apply_linked_review_contract(
            source,
            replayed_contract,
        )

        np.testing.assert_array_equal(first, second)

    def test_ui_and_managed_review_png_use_same_frozen_pixels(self):
        source = _review_scene()
        star_reference, star_config = _review_star_visibility_contract()
        contract = display_rendition.build_linked_review_contract(
            source,
            reason="bright_core_starless_rejected_after_recovery",
            source_stem="stage7_review_with_stars",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ui_path = root / "ui.png"
            write_display_preview(
                source,
                ui_path,
                apply_stretch=False,
                display_contract=contract,
            )
            report = export_managed_outputs(
                source,
                work_dir=root,
                base_filename="result_review",
                output_format="png",
                target_type="bright_emission_reflection_nebula",
                stars_required=True,
                star_reference=star_reference,
                star_visibility_config=star_config,
                display_contract=contract,
            )
            review = create_image_review_bundle(
                source,
                source,
                output_dir=root / "review_bundle",
                stage_key="stage7_stretching",
                source={
                    "before_stem": "stage6_input",
                    "after_stem": "stage7_review_with_stars",
                },
                display_contract=contract,
            )
            managed_path = root / "result_review_display_srgb.png"
            review_after_path = Path(review["previews"]["after_preview"])

            self.assertTrue(report["ready"], report)
            np.testing.assert_array_equal(
                _read_managed_display_png(ui_path),
                _read_managed_display_png(managed_path),
            )
            np.testing.assert_array_equal(
                _read_managed_display_png(review_after_path),
                _read_managed_display_png(managed_path),
            )
            self.assertEqual(
                report["display_visibility"]["transform"]["name"],
                display_rendition.CONTRACT_NAME,
            )

    def test_invalid_required_contract_does_not_publish_png(self):
        source = _review_scene()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = export_managed_outputs(
                source,
                work_dir=root,
                base_filename="result_review",
                output_format="png",
                display_contract={"status": "unavailable"},
            )

            self.assertFalse(report["ready"])
            self.assertFalse(
                (root / "result_review_display_srgb.png").exists()
            )
            self.assertTrue(
                any("display_png_failed" in issue for issue in report["issues"])
            )


if __name__ == "__main__":
    unittest.main()
