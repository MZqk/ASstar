#!/usr/bin/env python3
"""Stage 9 final delivery-contract and telemetry boundary tests."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


class PipelinePluginFallbackStage9DeliveryTests(PipelinePluginFallbackTestBase):
    def test_stage9_star_delivery_contract_accepts_only_formal_remixed_output(self):
        processor = self._new_processor()
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = True
        processor._stage9_output_contains_stars = True
        processor._stage9_output_withheld = False
        processor._stage9_remix_formally_accepted = True
        processor._stage9_final_source = "stage9_remixed"

        stage9_module = sys.modules["stages.stage9_star_remixing"]
        accepted = stage9_module._update_stage9_star_delivery_contract(
            processor
        )

        self.assertTrue(accepted)
        self.assertTrue(processor._stage9_star_delivery_contract_accepted)

        processor._stage9_final_source = "stage8_review_with_stars"
        accepted = stage9_module._update_stage9_star_delivery_contract(
            processor
        )
        self.assertFalse(accepted)

    def test_stage9_quality_report_separates_final_and_attempt_reason_codes(self):
        processor = self._new_processor()
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = True
        processor._stage9_output_contains_stars = True
        processor._stage9_output_withheld = False
        processor._stage9_remix_formally_accepted = True
        processor._stage9_final_source = "stage9_remixed"
        attempts = [
            {"attempt": "rejected", "reason_codes": ["old_attempt_failure"]},
            {
                "attempt": "selected",
                "accepted": True,
                "reason_codes": ["selected_advisory"],
            },
        ]

        stage9_module = sys.modules["stages.stage9_star_remixing"]
        stage9_module._write_stage9_quality_report(
            processor,
            attempts,
            attempts[-1],
            source_stem="stage8_enhanced",
            mode="screen",
        )

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["reason_codes_scope"], "attempt_history_legacy")
        self.assertIn("old_attempt_failure", report["attempt_history_reason_codes"])
        self.assertEqual(report["final_reason_codes"], ["selected_advisory"])
        self.assertTrue(report["star_delivery_contract_accepted"])


if __name__ == "__main__":
    unittest.main()
