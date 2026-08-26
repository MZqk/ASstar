#!/usr/bin/env python3
"""Non-visual state tests for the Stage processing-parameter accordion."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from tests.test_gui_runtime_modes import gui_module
from pipeline.processing_parameters import default_processing_parameters


class _Widget:
    def __init__(self, *, checked: bool = False):
        self.visible = True
        self.enabled = True
        self.checked = checked
        self.text = ""
        self.tooltip = ""
        self._blocked = False

    def setVisible(self, value):
        self.visible = bool(value)

    def setEnabled(self, value):
        self.enabled = bool(value)

    def setChecked(self, value):
        self.checked = bool(value)

    def isChecked(self):
        return self.checked

    def blockSignals(self, value):
        self._blocked = bool(value)

    def setText(self, value):
        self.text = str(value)

    def setToolTip(self, value):
        self.tooltip = str(value)


class ProcessingParameterUiStateTests(unittest.TestCase):
    def test_accordion_keeps_exactly_one_stage_open(self) -> None:
        headers = {stage: _Widget(checked=stage == 2) for stage in range(1, 11)}
        sections = {stage: _Widget() for stage in range(1, 11)}
        proxy = SimpleNamespace(
            _stage_parameter_headers=headers,
            _stage_parameter_sections=sections,
            _refresh_processing_stage_headers=lambda: None,
        )

        gui_module.StarunGui._toggle_processing_stage(proxy, 5, True)

        self.assertEqual(
            [stage for stage, section in sections.items() if section.visible],
            [5],
        )
        self.assertEqual(
            [stage for stage, header in headers.items() if header.checked],
            [5],
        )

    def test_hiding_expert_rows_does_not_clear_values(self) -> None:
        expert_widgets = {stage: [_Widget(), _Widget()] for stage in range(1, 11)}
        processing_parameters = {
            "stages": {
                "5": {"overrides": {"stage5_rl_iters": 12}}
            }
        }
        proxy = SimpleNamespace(
            processing_expert_visible=True,
            processing_expert_btn=_Widget(checked=True),
            _stage_expert_widgets=expert_widgets,
            _refresh_processing_stage_headers=lambda: None,
            _restoring_settings=True,
            processing_parameters=processing_parameters,
        )

        gui_module.StarunGui._set_processing_expert_visible(proxy, False)

        self.assertFalse(proxy.processing_expert_visible)
        self.assertTrue(
            all(not widget.visible for rows in expert_widgets.values() for widget in rows)
        )
        self.assertEqual(
            proxy.processing_parameters["stages"]["5"]["overrides"],
            {"stage5_rl_iters": 12},
        )

    def test_linear_resume_disables_stage1_through_stage5(self) -> None:
        headers = {stage: _Widget(checked=stage == 2) for stage in range(1, 11)}
        sections = {stage: _Widget() for stage in range(1, 11)}
        opened: list[int] = []
        proxy = SimpleNamespace(
            processing_color_group=_Widget(),
            processing_stage_groups=sections,
            _stage_parameter_headers=headers,
            processing_sheet_note=_Widget(),
            _current_input_mode=lambda: gui_module.INPUT_MODE_LINEAR_RESUME,
            _toggle_processing_stage=lambda stage, checked: opened.append(stage),
        )

        gui_module.StarunGui._update_processing_sheet_availability(proxy)

        self.assertTrue(all(not sections[stage].enabled for stage in range(1, 6)))
        self.assertTrue(all(sections[stage].enabled for stage in range(6, 11)))
        self.assertEqual(opened, [6])
        self.assertIn("Stage 1–5", proxy.processing_sheet_note.text)

    def test_stage1_resume_disables_only_stage1(self) -> None:
        headers = {stage: _Widget(checked=stage == 1) for stage in range(1, 11)}
        sections = {stage: _Widget() for stage in range(1, 11)}
        opened: list[int] = []
        proxy = SimpleNamespace(
            processing_color_group=_Widget(),
            processing_stage_groups=sections,
            _stage_parameter_headers=headers,
            processing_sheet_note=_Widget(),
            _current_input_mode=lambda: "stage1_prepared_resume",
            _toggle_processing_stage=lambda stage, checked: opened.append(stage),
        )

        gui_module.StarunGui._update_processing_sheet_availability(proxy)

        self.assertFalse(sections[1].enabled)
        self.assertTrue(all(sections[stage].enabled for stage in range(2, 11)))
        self.assertEqual(opened, [2])

    def test_stage_header_marks_hidden_expert_override(self) -> None:
        headers = {stage: _Widget(checked=stage == 5) for stage in range(1, 11)}
        payload = default_processing_parameters()
        payload["stages"]["5"]["overrides"][
            "stage5_multiscale_detail_retention_min"
        ] = 0.90
        proxy = SimpleNamespace(
            _stage_parameter_headers=headers,
            processing_parameters=payload,
            processing_expert_visible=False,
        )

        gui_module.StarunGui._refresh_processing_stage_headers(proxy)

        self.assertIn("1 项自定义", headers[5].text)
        self.assertIn("专家配置已生效", headers[5].text)

    def test_stage8_target_aware_chroma_dependency_follows_full_saturation(
        self,
    ) -> None:
        field = "stage8_target_aware_chroma_enabled"
        control = _Widget()
        auto_check = _Widget()
        row_label = _Widget()
        payload = default_processing_parameters()
        proxy = SimpleNamespace(
            _stage_parameter_controls={field: control},
            processing_parameters=payload,
            _stage_parameter_effective_labels={},
            _stage_parameter_auto_checks={field: auto_check},
            _stage_parameter_row_widgets={field: (row_label, _Widget())},
        )

        gui_module.StarunGui._refresh_processing_parameter_dependencies(proxy)
        self.assertTrue(control.enabled)
        self.assertTrue(auto_check.enabled)
        self.assertTrue(row_label.enabled)

        payload["stages"]["8"]["mode"] = "limited"
        gui_module.StarunGui._refresh_processing_parameter_dependencies(proxy)
        self.assertFalse(control.enabled)
        self.assertFalse(auto_check.enabled)
        self.assertFalse(row_label.enabled)

        payload["stages"]["8"]["mode"] = "auto"
        payload["stages"]["8"]["overrides"][
            "stage8_nebula_saturation_enabled"
        ] = False
        gui_module.StarunGui._refresh_processing_parameter_dependencies(proxy)
        self.assertFalse(control.enabled)

        payload["stages"]["8"]["overrides"][
            "stage8_nebula_saturation_enabled"
        ] = True
        gui_module.StarunGui._refresh_processing_parameter_dependencies(proxy)
        self.assertTrue(control.enabled)


if __name__ == "__main__":
    unittest.main()
