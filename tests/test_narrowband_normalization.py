from __future__ import annotations

import sys
import unittest
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from narrowband_normalization import (  # noqa: E402
    resolve_dual_narrowband_mapping,
)


class NarrowbandNormalizationTests(unittest.TestCase):
    def test_mapping_requires_identified_emission_lines(self) -> None:
        explicit = resolve_dual_narrowband_mapping(
            {"FILTER": "Ha + OIII dual-band"}
        )
        generic = resolve_dual_narrowband_mapping(
            {"FILTER": "generic dualband"}
        )
        unknown = resolve_dual_narrowband_mapping({})

        self.assertGreaterEqual(explicit["confidence"], 0.85)
        self.assertEqual(generic["confidence"], 0.86)
        self.assertEqual(generic["evidence"], "generic_dualband_hint")
        self.assertEqual(unknown["mapping"], "unknown")

    def test_verified_starun_profile_matches_contract(self) -> None:
        mapping = resolve_dual_narrowband_mapping(
            {"INSTRUME": "Seestar S30 Pro", "FILTER": "LP"}
        )

        self.assertEqual(mapping["mapping"], "osc_hoo_rgb")
        self.assertEqual(mapping["ha_channel"], "R")
        self.assertEqual(mapping["oiii_channels"], ["G", "B"])
        self.assertEqual(mapping["confidence"], 0.97)
        self.assertEqual(mapping["evidence"], "verified_device_profile")
        self.assertEqual(
            mapping["evidence_detail"]["device_profile_id"],
            "seestar_s30_pro_imx585",
        )

    def test_generic_and_conflicting_filter_evidence(self) -> None:
        generic = resolve_dual_narrowband_mapping(
            {"INSTRUME": "Other OSC", "FILTER": "Duo-Band"}
        )
        bare_lp = resolve_dual_narrowband_mapping(
            {"INSTRUME": "Other OSC", "FILTER": "LP"}
        )
        conflict = resolve_dual_narrowband_mapping(
            {"INSTRUME": "Other OSC", "FILTER": "SII/OIII Dual-Band"}
        )

        self.assertEqual(generic["confidence"], 0.86)
        self.assertEqual(generic["evidence"], "generic_dualband_hint")
        self.assertEqual(bare_lp["mapping"], "osc_hoo_rgb")
        self.assertEqual(bare_lp["confidence"], 0.86)
        self.assertEqual(
            bare_lp["evidence"],
            "authoritative_filter_field_hint",
        )
        self.assertEqual(conflict["mapping"], "unknown")
        self.assertEqual(conflict["evidence"], "conflicting_filter_lines")

        dwarf2 = resolve_dual_narrowband_mapping(
            {"INSTRUME": "DWARF 2", "FILTER": "Dual-Band"}
        )
        self.assertEqual(dwarf2["confidence"], 0.86)
        self.assertEqual(dwarf2["evidence"], "generic_dualband_hint")
        self.assertNotEqual(dwarf2["evidence"], "verified_device_profile")

        tri_line_conflict = resolve_dual_narrowband_mapping(
            {"FILTER": "Ha / OIII / SII"}
        )
        self.assertEqual(tri_line_conflict["mapping"], "unknown")
        self.assertEqual(
            tri_line_conflict["evidence"],
            "conflicting_filter_lines",
        )

    def test_supported_device_aliases_and_wide_path_guard(self) -> None:
        cases = (
            ("ZWO Seestar S30", "LP"),
            ("Seestar S30", "LP_Starless"),
            ("S30", "lp"),
            ("ZWO Seestar S30 Pro", "LP_Starless"),
            ("Seestar S30 Pro", "lp"),
            ("S30 Pro", "LP"),
            ("ZWO Seestar S50", "LP_Starless"),
            ("Seestar S50", "LP"),
            ("S50", "lp"),
            ("DWARFLAB DWARF 3", "Dual-Band"),
            ("DWARF 3", "Duo-Band"),
            ("DWARF III", "duoband"),
            ("DWARFIII", "dualband"),
            ("DWARF3", "dual-band"),
            ("DWARFLAB DWARF mini", "Duo-Band"),
            ("DWARF mini", "Dual-Band"),
            ("DWARFmini", "duoband"),
        )
        for instrument, filter_name in cases:
            with self.subTest(instrument=instrument, filter_name=filter_name):
                mapping = resolve_dual_narrowband_mapping(
                    {
                        "instrume": instrument,
                        "filter": (f" {filter_name}   ", "filter comment"),
                    }
                )
                self.assertEqual(mapping["confidence"], 0.97)
                self.assertEqual(mapping["evidence"], "verified_device_profile")

        insflnam = resolve_dual_narrowband_mapping(
            {
                "INSTRUME": "ZWO Seestar S30 Pro",
                "INSFLNAM": (" LP_Starless   ", "filter comment"),
            }
        )
        self.assertEqual(insflnam["confidence"], 0.97)
        self.assertEqual(
            insflnam["evidence_detail"]["header_key"],
            "INSFLNAM",
        )

        wide = resolve_dual_narrowband_mapping(
            {
                "INSTRUME": "Seestar S30 Pro",
                "SENSOR": "Sony IMX586",
                "FOCALLEN": 6.0,
                "FILTER": "LP",
            }
        )
        self.assertEqual(wide["mapping"], "unknown")
        self.assertEqual(wide["evidence"], "wide_path_not_supported")

    def test_filter_products_are_verified_without_object_inference(self) -> None:
        for filter_name in (
            "ZWO Duo-Band",
            "Optolong L-eXtreme",
            "Optolong L-Ultimate",
            "Optolong L-Para",
            "IDAS NBZ",
            "IDAS NBZ-II",
            "SVBONY SV220",
        ):
            with self.subTest(filter_name=filter_name):
                mapping = resolve_dual_narrowband_mapping(
                    {"INSTRUME": "Generic OSC", "FILTER": filter_name}
                )
                self.assertEqual(mapping["confidence"], 0.93)
                self.assertEqual(mapping["evidence"], "verified_filter_profile")
        object_only = resolve_dual_narrowband_mapping(
            {"OBJECT": "HOO test target"}
        )
        self.assertEqual(object_only["mapping"], "unknown")

    def test_filter_is_authoritative_and_supplemental_fallback_is_audited(self) -> None:
        authoritative_lp = resolve_dual_narrowband_mapping(
            {"FILTER": " LP_Starless   ", "FILTER1": "SII"}
        )
        authoritative_broadband = resolve_dual_narrowband_mapping(
            {"FILTER": "IRCUT", "FILTER1": "Dual-Band"}
        )
        fallback = resolve_dual_narrowband_mapping(
            {"FILTER": "Custom", "FILTER1": "L-eXtreme"}
        )
        fallback_conflict = resolve_dual_narrowband_mapping(
            {
                "FILTER": "Custom",
                "FILTER1": "Clear",
                "FILTER2": "Dual-Band",
            }
        )

        self.assertEqual(authoritative_lp["mapping"], "osc_hoo_rgb")
        self.assertEqual(authoritative_lp["confidence"], 0.86)
        self.assertEqual(
            authoritative_lp["evidence"],
            "authoritative_filter_field_hint",
        )
        lp_detail = authoritative_lp["evidence_detail"]
        self.assertEqual(lp_detail["selection_source"], "authoritative_filter")
        self.assertEqual(
            lp_detail["selected_filter_headers"][0]["header_key"],
            "FILTER",
        )
        self.assertEqual(
            lp_detail["ignored_filter_headers"][0]["header_key"],
            "FILTER1",
        )
        self.assertEqual(authoritative_broadband["mapping"], "unknown")
        self.assertEqual(
            authoritative_broadband["evidence"],
            "explicit_broadband_filter",
        )
        self.assertEqual(fallback["confidence"], 0.93)
        self.assertEqual(fallback["evidence"], "verified_filter_profile")
        self.assertEqual(
            fallback["evidence_detail"]["selection_source"],
            "supplemental_fallback",
        )
        self.assertEqual(fallback_conflict["mapping"], "unknown")
        self.assertEqual(
            fallback_conflict["evidence"],
            "conflicting_filter_fields",
        )

    def test_supplemental_filter_field_aliases_are_supported(self) -> None:
        for key in (
            "FILTER1",
            "FILTER2",
            "INSFLNAM",
            "FILTERNAME",
            "FILTNAME",
            "FILTNAM",
        ):
            with self.subTest(key=key):
                mapping = resolve_dual_narrowband_mapping(
                    {"FILTER": "Custom", key.lower(): (" L-eXtreme   ", "comment")}
                )
                self.assertEqual(mapping["confidence"], 0.93)
                self.assertEqual(
                    mapping["evidence_detail"]["selected_filter_headers"][0][
                        "header_key"
                    ],
                    key,
                )

    def test_explicit_broadband_aliases_stop_supplemental_inference(self) -> None:
        for filter_name in (
            "IRCUT",
            "IR Cut",
            "UV-IR",
            "Astro",
            "VIS",
            "Clear",
            "No LP",
            "LP Off",
        ):
            with self.subTest(filter_name=filter_name):
                mapping = resolve_dual_narrowband_mapping(
                    {"FILTER": filter_name, "FILTER1": "Dual-Band"}
                )
                self.assertEqual(mapping["mapping"], "unknown")
                self.assertEqual(
                    mapping["evidence"],
                    "explicit_broadband_filter",
                )

    def test_explicit_user_hint_only_fills_missing_or_ambiguous_header(self) -> None:
        for metadata in ({}, {"FILTER": "Custom"}):
            with self.subTest(metadata=metadata):
                mapping = resolve_dual_narrowband_mapping(
                    metadata,
                    filter_hint="dualband Ha OIII",
                )
                self.assertEqual(mapping["confidence"], 0.99)
                self.assertEqual(mapping["evidence"], "explicit_user_hint")

        authoritative_lp = resolve_dual_narrowband_mapping(
            {"FILTER": "LP"},
            filter_hint="dualband Ha OIII",
        )
        generic_dualband = resolve_dual_narrowband_mapping(
            {"FILTER": "Duo-Band"},
            filter_hint="dualband Ha OIII",
        )
        self.assertEqual(authoritative_lp["confidence"], 0.86)
        self.assertEqual(
            authoritative_lp["evidence"],
            "authoritative_filter_field_hint",
        )
        self.assertEqual(generic_dualband["confidence"], 0.86)
        self.assertEqual(generic_dualband["evidence"], "generic_dualband_hint")

        single_line = resolve_dual_narrowband_mapping(
            {"FILTER": "OIII 3nm"},
            filter_hint="dualband Ha OIII",
        )
        broadband = resolve_dual_narrowband_mapping(
            {"FILTER": "UV/IR Cut"},
            filter_hint="dualband Ha OIII",
        )
        self.assertEqual(single_line["mapping"], "unknown")
        self.assertEqual(broadband["mapping"], "unknown")


if __name__ == "__main__":
    unittest.main()
