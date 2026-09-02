#!/usr/bin/env python3
"""Runtime capability manifest and Gaia preflight regression tests."""

from __future__ import annotations

import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from gui import common
from gui import native_pipeline_runtime as native_runtime
from gui import runtime_capabilities as capabilities
from tests.test_native_pipeline_runtime import (
    fake_codesign_metadata,
    fake_unsigned_macho_sha256,
    make_test_native_payload,
)


class _Response:
    def __init__(self, status: int = 206) -> None:
        self.status = status
        self.closed = False

    def read(self, _size: int = -1) -> bytes:
        return b"x"

    def close(self) -> None:
        self.closed = True


class RuntimeCapabilitiesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._codesign_patcher = patch.object(
            native_runtime,
            "_codesign_metadata",
            side_effect=fake_codesign_metadata,
        )
        self._unsigned_macho_patcher = patch.object(
            native_runtime,
            "unsigned_macho_sha256",
            side_effect=fake_unsigned_macho_sha256,
        )
        self._codesign_patcher.start()
        self._unsigned_macho_patcher.start()
        self.addCleanup(self._unsigned_macho_patcher.stop)
        self.addCleanup(self._codesign_patcher.stop)

    def _make_bundle_layout(
        self,
        root: Path,
        *,
        formal_app: bool = True,
    ) -> dict[str, Path]:
        resources = (
            root / "Installed Starun.app" / "Contents" / "Resources"
            if formal_app
            else root / "development-overlay"
        )
        resources.mkdir(parents=True)

        siril = resources / "Siril.app" / "Contents" / "MacOS" / "siril-cli"
        siril.parent.mkdir(parents=True)
        siril.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        siril.chmod(0o755)

        template = resources / "config.1.4.ini.template"
        template.write_text("[core]\nextension=.fit\n", encoding="utf-8")

        pipeline = resources / "pipeline" / "starun.py"
        pipeline.parent.mkdir(parents=True)
        pipeline.write_text("VALUE = 1\n", encoding="utf-8")
        for relative in capabilities.PIPELINE_REQUIRED_PATHS:
            path = pipeline.parent / relative
            if path.suffix == ".py":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# runtime component\n", encoding="utf-8")
            else:
                path.mkdir(parents=True, exist_ok=True)
        if formal_app:
            make_test_native_payload(pipeline.parent)
        else:
            for name in native_runtime.NATIVE_MODULES:
                (pipeline.parent / f"{name}.py").write_text(
                    "# development source module\n",
                    encoding="utf-8",
                )

        plugins = resources / "siril_plugins"
        plugins.mkdir()
        runtime_home = root / "runtime-home"
        runtime_home.mkdir()
        return {
            "resources": resources,
            "siril": siril,
            "template": template,
            "pipeline": pipeline,
            "plugins": plugins,
            "runtime_home": runtime_home,
        }

    def _manifest(
        self,
        layout: dict[str, Path],
        *,
        network_enabled: bool,
        fallback_mode: str = "auto_local_reference",
        pipeline: Path | None = None,
        endpoints=None,
    ) -> dict[str, object]:
        return capabilities.build_runtime_capabilities(
            resources_root=layout["resources"],
            runtime_home=layout["runtime_home"],
            siril_candidates=[layout["siril"]],
            config_template=layout["template"],
            pipeline_path=pipeline or layout["pipeline"],
            siril_plugin_dir=layout["plugins"],
            network_enabled=network_enabled,
            stage4_offline_fallback_mode=fallback_mode,
            run_id="run-1",
            endpoints=endpoints,
        )

    def _install_complete_local_catalogs(self, runtime_home: Path) -> None:
        astro, xp_root = capabilities.runtime_catalog_paths(runtime_home)
        astro.parent.mkdir(parents=True, exist_ok=True)
        with astro.open("wb") as handle:
            handle.truncate(capabilities.GAIA_ASTRO_EXPECTED_SIZE_BYTES)
        xp_root.mkdir()
        for index in range(capabilities.GAIA_XP_EXPECTED_CHUNKS):
            (xp_root / f"{capabilities.GAIA_XP_FILE_PREFIX}{index}.dat").write_bytes(
                b"x" * capabilities.GAIA_XP_MIN_CHUNK_BYTES
            )

    def test_default_xp_preflight_uses_only_zenodo(self) -> None:
        expected = (
            "https://zenodo.org/records/17988559/files/"
            "siril_cat1_healpix8_xpsamp_0.dat",
        )

        self.assertEqual(capabilities.DEFAULT_GAIA_XP_ENDPOINTS, expected)
        self.assertEqual(
            capabilities.configured_network_endpoints({})["gaia_xp"],
            expected,
        )

    def test_manifest_uses_actual_bundle_and_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            layout = self._make_bundle_layout(Path(td))
            manifest = self._manifest(layout, network_enabled=True)

            origin = manifest["resource_origin"]
            self.assertEqual(origin["kind"], "app_bundle")
            self.assertEqual(
                Path(origin["resources_root"]),
                layout["resources"].resolve(),
            )
            self.assertEqual(
                Path(manifest["runtime_home"]),
                layout["runtime_home"].resolve(),
            )
            rendered = str(manifest)
            self.assertNotIn("release/Starun.app", rendered)
            self.assertEqual(manifest["blocking_errors"], [])

    def test_bundle_boundary_rejects_pipeline_from_another_app(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            layout = self._make_bundle_layout(root)
            external_layout = self._make_bundle_layout(root / "release")

            manifest = self._manifest(
                layout,
                network_enabled=True,
                pipeline=external_layout["pipeline"],
            )

            pipeline = manifest["capabilities"]["pipeline"]
            self.assertFalse(pipeline["within_resources_root"])
            self.assertFalse(pipeline["available"])
            self.assertTrue(
                any("流水线资源" in error for error in manifest["blocking_errors"])
            )

    def test_formal_app_requires_a_valid_native_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            layout = self._make_bundle_layout(Path(td))
            ready = self._manifest(layout, network_enabled=True)
            native = ready["capabilities"]["native_pipeline"]
            self.assertTrue(native["required"])
            self.assertTrue(native["available"])
            self.assertEqual(native["mode"], "native")

            (layout["pipeline"].parent / native_runtime.NATIVE_RUNTIME_MANIFEST_NAME).unlink()
            missing = self._manifest(layout, network_enabled=True)
            missing_native = missing["capabilities"]["native_pipeline"]
            self.assertFalse(missing_native["available"])
            self.assertEqual(missing_native["mode"], "native_missing")
            self.assertFalse(missing["capabilities"]["pipeline"]["available"])
            self.assertTrue(
                any("原生流水线不可用" in error for error in missing["blocking_errors"])
            )

    def test_development_overlay_allows_complete_source_modules(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            layout = self._make_bundle_layout(
                Path(td),
                formal_app=False,
            )

            manifest = self._manifest(layout, network_enabled=True)

            native = manifest["capabilities"]["native_pipeline"]
            self.assertFalse(native["required"])
            self.assertTrue(native["available"])
            self.assertEqual(native["mode"], "source")
            self.assertTrue(manifest["capabilities"]["pipeline"]["available"])
            self.assertEqual(manifest["blocking_errors"], [])

    def test_invalid_development_manifest_does_not_fallback_to_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            layout = self._make_bundle_layout(
                Path(td),
                formal_app=False,
            )
            (layout["pipeline"].parent / native_runtime.NATIVE_RUNTIME_MANIFEST_NAME).write_text(
                "{}\n",
                encoding="utf-8",
            )

            manifest = self._manifest(layout, network_enabled=True)

            native = manifest["capabilities"]["native_pipeline"]
            self.assertFalse(native["available"])
            self.assertEqual(native["mode"], "native_invalid")
            self.assertFalse(manifest["capabilities"]["pipeline"]["available"])
            self.assertTrue(manifest["blocking_errors"])

    def test_new_pipeline_modules_are_required_by_build_and_preflight(self) -> None:
        required_modules = (
            "final_artifact_identity.py",
            "presentation_quality.py",
            "spatial_background_lineage.py",
            "stage5_handoff.py",
            "stage8_handoff.py",
            "stage8_color_rendition.py",
            "stage8_starless_finish.py",
            "star_halo_guard.py",
        )
        build_script = (
            Path(__file__).resolve().parents[1] / "build" / "build_macos_app.sh"
        ).read_text(encoding="utf-8")
        required_block = build_script.split(
            "PIPELINE_REQUIRED_MODULES=(", 1
        )[1].split(")", 1)[0]
        for module_name in required_modules:
            self.assertIn(module_name, capabilities.PIPELINE_REQUIRED_PATHS)
            self.assertIn(module_name, required_block)

        for module_name in required_modules:
            with self.subTest(module=module_name), tempfile.TemporaryDirectory() as td:
                layout = self._make_bundle_layout(Path(td))
                (layout["pipeline"].parent / module_name).unlink()

                manifest = self._manifest(layout, network_enabled=True)

                pipeline = manifest["capabilities"]["pipeline"]
                self.assertFalse(pipeline["available"])
                self.assertIn(module_name, pipeline["missing_required_paths"])
                self.assertTrue(
                    any(
                        "流水线资源" in error
                        for error in manifest["blocking_errors"]
                    )
                )

    def test_offline_mode_allows_degraded_route_then_uses_complete_local_gaia(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            layout = self._make_bundle_layout(Path(td))
            missing = self._manifest(layout, network_enabled=False)
            decision = missing["decisions"]["stage4_color_calibration"]
            self.assertEqual(missing["blocking_errors"], [])
            self.assertEqual(missing["status"], "degraded_allowed")
            self.assertEqual(decision["route"], "auto_local_reference")
            self.assertEqual(
                decision["skip_photometric_commands"],
                ["platesolve", "spcc", "pcc"],
            )
            self.assertTrue(decision["requires_review"])

            self._install_complete_local_catalogs(layout["runtime_home"])
            complete = self._manifest(layout, network_enabled=False)
            self.assertEqual(complete["blocking_errors"], [])
            self.assertEqual(complete["status"], "ready")
            self.assertTrue(complete["capabilities"]["gaia_astro"]["available"])
            self.assertEqual(
                complete["capabilities"]["gaia_xp"]["valid_chunk_count"],
                48,
            )
            complete_decision = complete["decisions"][
                "stage4_color_calibration"
            ]
            self.assertEqual(complete_decision["spcc_readiness"], "local_verified")
            self.assertTrue(complete_decision["spcc_operational_verified"])

    def test_online_mode_accepts_reachable_endpoint_group_as_local_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            layout = self._make_bundle_layout(Path(td))
            endpoint_map = {
                "gaia_astro": ("https://astro.example.test/availability",),
                "gaia_xp": (
                    "https://failed.example.test/chunk.dat",
                    "https://xp.example.test/chunk.dat",
                ),
            }
            manifest = self._manifest(
                layout,
                network_enabled=True,
                endpoints=endpoint_map,
            )
            capabilities.update_siril_launch_probe(
                manifest,
                launchable=True,
                version="siril 1.4.4",
            )
            solver_backends = manifest["capabilities"][
                "stage4_plate_solver_backends"
            ]
            self.assertEqual(
                [item["id"] for item in solver_backends],
                list(capabilities.STAGE4_PLATE_SOLVER_BACKEND_IDS),
            )
            self.assertEqual(solver_backends[0]["version"], "siril 1.4.4")
            self.assertTrue(solver_backends[0]["eligible"])
            self.assertTrue(
                all(
                    item["runtime_status"] == "not_probed"
                    for item in solver_backends[1:]
                )
            )

            def opener(request, *, timeout):
                self.assertGreater(timeout, 0)
                if "failed.example.test" in request.full_url:
                    raise urllib.error.URLError("offline")
                return _Response()

            capabilities.probe_network_capabilities(manifest, opener=opener)

            self.assertEqual(manifest["blocking_errors"], [])
            self.assertEqual(manifest["status"], "ready")
            groups = manifest["capabilities"]["network_endpoints"]["groups"]
            self.assertTrue(groups["gaia_astro"]["available"])
            self.assertTrue(groups["gaia_xp"]["available"])
            self.assertEqual(len(groups["gaia_xp"]["probes"]), 2)
            self.assertEqual(
                groups["gaia_xp"]["evidence_level"],
                "endpoint_reachability_only",
            )
            self.assertFalse(groups["gaia_xp"]["service_verified"])
            decision = manifest["decisions"]["stage4_color_calibration"]
            self.assertEqual(decision["spcc_readiness"], "online_unverified")
            self.assertFalse(decision["spcc_operational_verified"])
            self.assertEqual(decision["spcc_online_unverified_timeout_sec"], 300)
            self.assertIn(
                "gaia_xp_endpoint_reachable_spcc_unverified",
                decision["reason_codes"],
            )
            self.assertTrue(
                any("单次预算 300 秒" in line for line in capabilities.capability_summary_lines(manifest))
            )

    def test_online_spcc_operational_cache_key_tracks_siril_and_xp_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            layout = self._make_bundle_layout(Path(td))
            manifest = self._manifest(
                layout,
                network_enabled=True,
                endpoints={
                    "gaia_astro": ("https://astro.example.test/availability",),
                    "gaia_xp": ("https://xp.example.test/chunk.dat",),
                },
            )
            capabilities.update_siril_launch_probe(
                manifest,
                launchable=True,
                version="siril 1.4.4",
            )
            capabilities.probe_network_capabilities(
                manifest,
                opener=lambda *_args, **_kwargs: _Response(),
            )

            cache_key = capabilities.stage4_spcc_operational_cache_key(manifest)
            self.assertIsNotNone(cache_key)
            manifest["generated_at"] = "later"
            manifest["run_id"] = "another-run"
            self.assertEqual(
                capabilities.stage4_spcc_operational_cache_key(manifest),
                cache_key,
            )

            launch_probe = manifest["capabilities"]["siril"]["launch_probe"]
            launch_probe["version"] = "siril 1.4.5"
            self.assertNotEqual(
                capabilities.stage4_spcc_operational_cache_key(manifest),
                cache_key,
            )

    def test_cached_online_spcc_timeout_routes_to_pcc_with_audit_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            layout = self._make_bundle_layout(Path(td))
            manifest = self._manifest(
                layout,
                network_enabled=True,
                endpoints={
                    "gaia_astro": ("https://astro.example.test/availability",),
                    "gaia_xp": ("https://xp.example.test/chunk.dat",),
                },
            )
            capabilities.update_siril_launch_probe(
                manifest,
                launchable=True,
                version="siril 1.4.4",
            )
            capabilities.probe_network_capabilities(
                manifest,
                opener=lambda *_args, **_kwargs: _Response(),
            )
            cache_key = capabilities.stage4_spcc_operational_cache_key(manifest)
            assert cache_key is not None

            capabilities.apply_stage4_spcc_operational_timeout_cache(
                manifest,
                cache_key=cache_key,
                evidence={"status": "timeout", "timeout_sec": 90},
            )

            decision = manifest["decisions"]["stage4_color_calibration"]
            self.assertEqual(manifest["status"], "degraded_allowed")
            self.assertEqual(decision["route"], "physical_pcc_only")
            self.assertFalse(decision["commands"]["spcc"])
            self.assertTrue(decision["commands"]["pcc"])
            self.assertEqual(
                decision["spcc_operational_cache"]["status"],
                "operational_timeout_cached",
            )
            self.assertIn("operational_timeout_cached", decision["reason_codes"])

    def test_transient_exhaustion_uses_v2_operational_cache_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            layout = self._make_bundle_layout(Path(td))
            manifest = self._manifest(
                layout,
                network_enabled=True,
                endpoints={
                    "gaia_astro": ("https://astro.example.test/availability",),
                    "gaia_xp": ("https://xp.example.test/chunk.dat",),
                },
            )
            capabilities.update_siril_launch_probe(
                manifest,
                launchable=True,
                version="siril 1.4.4",
            )
            capabilities.probe_network_capabilities(
                manifest,
                opener=lambda *_args, **_kwargs: _Response(),
            )
            cache_key = capabilities.stage4_spcc_operational_cache_key(manifest)
            assert cache_key is not None

            capabilities.apply_stage4_spcc_operational_timeout_cache(
                manifest,
                cache_key=cache_key,
                evidence={
                    "status": "online_transient_exhausted",
                    "failure_class": "transient_network",
                },
            )

            cache = manifest["decisions"]["stage4_color_calibration"][
                "spcc_operational_cache"
            ]
            self.assertEqual(
                cache["schema"],
                capabilities.SPCC_OPERATIONAL_CACHE_SCHEMA,
            )
            self.assertEqual(
                cache["status"],
                "operational_transient_failure_cached",
            )

    def test_online_transport_failure_remains_operationally_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            layout = self._make_bundle_layout(Path(td))
            endpoint_map = {
                "gaia_astro": ("https://astro.example.test/availability",),
                "gaia_xp": ("https://xp.example.test/chunk.dat",),
            }
            manifest = self._manifest(
                layout,
                network_enabled=True,
                endpoints=endpoint_map,
            )
            capabilities.update_siril_launch_probe(manifest, launchable=True)

            def opener(_request, *, timeout):
                self.assertGreater(timeout, 0)
                raise urllib.error.URLError("offline")

            capabilities.probe_network_capabilities(manifest, opener=opener)

            self.assertEqual(manifest["blocking_errors"], [])
            self.assertEqual(manifest["status"], "ready")
            decision = manifest["decisions"]["stage4_color_calibration"]
            self.assertEqual(decision["route"], "physical_spcc_then_pcc")
            self.assertTrue(decision["physical_color_available"])
            self.assertEqual(decision["spcc_readiness"], "online_unverified")
            self.assertTrue(decision["commands"]["platesolve"])
            self.assertTrue(decision["commands"]["spcc"])
            self.assertTrue(decision["commands"]["pcc"])
            self.assertIn(
                "gaia_astrometry_endpoint_unreachable_operational_probe_required",
                decision["reason_codes"],
            )

    def test_online_confirmed_http_failure_is_advisory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            layout = self._make_bundle_layout(Path(td))
            manifest = self._manifest(
                layout,
                network_enabled=True,
                endpoints={
                    "gaia_astro": ("https://astro.example.test/availability",),
                    "gaia_xp": ("https://xp.example.test/chunk.dat",),
                },
            )
            capabilities.update_siril_launch_probe(manifest, launchable=True)

            def opener(request, *, timeout):
                self.assertGreater(timeout, 0)
                self.assertTrue(request.full_url.startswith("https://"))
                return _Response(status=503)

            capabilities.probe_network_capabilities(manifest, opener=opener)

            decision = manifest["decisions"]["stage4_color_calibration"]
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(decision["route"], "physical_spcc_then_pcc")
            self.assertEqual(decision["attempt_policy"], "attempt_then_fallback")
            self.assertTrue(decision["preflight_advisory_only"])
            self.assertTrue(decision["physical_color_available"])
            self.assertTrue(decision["commands"]["spcc"])
            self.assertTrue(decision["commands"]["pcc"])

    def test_preserve_fallback_is_degraded_passthrough_not_a_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            layout = self._make_bundle_layout(Path(td))
            manifest = self._manifest(
                layout,
                network_enabled=False,
                fallback_mode="preserve",
            )

            self.assertEqual(manifest["blocking_errors"], [])
            self.assertEqual(manifest["status"], "degraded_allowed")
            decision = manifest["decisions"]["stage4_color_calibration"]
            self.assertEqual(decision["route"], "preserve_input")
            self.assertTrue(decision["preserve_input_available"])
            self.assertFalse(decision["auto_local_reference_available"])

    def test_local_astrometric_catalog_routes_to_pcc_only_without_xp(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            layout = self._make_bundle_layout(Path(td))
            astro, _xp = capabilities.runtime_catalog_paths(layout["runtime_home"])
            astro.parent.mkdir(parents=True, exist_ok=True)
            with astro.open("wb") as handle:
                handle.truncate(capabilities.GAIA_ASTRO_EXPECTED_SIZE_BYTES)

            manifest = self._manifest(layout, network_enabled=False)

            self.assertEqual(manifest["status"], "degraded_allowed")
            decision = manifest["decisions"]["stage4_color_calibration"]
            self.assertEqual(decision["route"], "physical_pcc_only")
            self.assertEqual(decision["astrometric_source"], "localgaia")
            self.assertEqual(decision["skip_photometric_commands"], ["spcc"])
            self.assertTrue(decision["pcc_available"])

    def test_explicit_overlay_resolves_pipeline_and_plugins_from_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            resources = Path(td) / "overlay"
            pipeline = resources / "pipeline" / "starun.py"
            plugins = resources / "siril_plugins"
            pipeline.parent.mkdir(parents=True)
            pipeline.write_text("# overlay pipeline\n", encoding="utf-8")
            plugins.mkdir()

            self.assertEqual(common.default_pipeline_path(resources), pipeline)
            self.assertEqual(common.default_siril_plugin_dir(resources), plugins)

    def test_frozen_resource_root_is_derived_from_running_executable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            executable = (
                Path(td)
                / "Mounted Starun.app"
                / "Contents"
                / "MacOS"
                / "Starun"
            )
            executable.parent.mkdir(parents=True)
            executable.write_text("binary", encoding="utf-8")
            with patch.object(common, "is_frozen", return_value=True), patch.object(
                common.sys,
                "executable",
                str(executable),
            ):
                resolved = common.resource_root()

            self.assertEqual(
                resolved,
                executable.resolve().parent.parent / "Resources",
            )
            self.assertNotIn("release/Starun.app", str(resolved))


if __name__ == "__main__":
    unittest.main()
