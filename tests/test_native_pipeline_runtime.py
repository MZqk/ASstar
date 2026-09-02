#!/usr/bin/env python3
"""Fail-closed native pipeline runtime contract tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build import manage_native_pipeline_bundle as native_manager
from gui import native_pipeline_runtime as native


TEST_TEAM_IDENTIFIER = "TESTTEAM1"


def fake_unsigned_macho_sha256(path: Path) -> str:
    return hashlib.sha256(b"unsigned-macho\0" + Path(path).read_bytes()).hexdigest()


def fake_codesign_metadata(path: Path) -> dict[str, object]:
    return {
        "verified": True,
        "cdhash": hashlib.sha256(Path(path).read_bytes()).hexdigest()[:40],
        "team_identifier": TEST_TEAM_IDENTIFIER,
        "hardened_runtime": True,
        "mode": "developer_id",
        "flags_detail": "runtime",
    }


def write_test_native_manifest(
    root: Path,
    payload: dict[str, object],
) -> Path:
    """Write the same canonical self-hash shape used by the runtime."""

    unsigned = dict(payload)
    unsigned.pop("manifest_payload_sha256", None)
    payload = dict(unsigned)
    payload["manifest_payload_sha256"] = native.canonical_payload_sha256(
        unsigned
    )
    path = root / native.NATIVE_RUNTIME_MANIFEST_NAME
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def make_test_native_payload(
    root: Path,
    *,
    marker: str = "native",
) -> dict[str, object]:
    """Create a byte-level fixture; Mach-O execution is covered by E2E."""

    root.mkdir(parents=True, exist_ok=True)
    build_manifest = root / native.NATIVE_BUILD_MANIFEST_NAME
    build_manifest.write_bytes(b'{"schema":"build-provenance"}\n')
    modules: list[dict[str, object]] = []
    for index, name in enumerate(native.NATIVE_MODULES):
        filename = f"{name}{native.NATIVE_EXTENSION_SUFFIX}"
        binary = root / filename
        binary.write_bytes(f"{marker}-{index}-{name}".encode("utf-8"))
        modules.append(
            {
                "import_name": name,
                "binary_filename": filename,
                "binary_sha256": native.sha256_file(binary),
                "unsigned_macho_sha256": fake_unsigned_macho_sha256(binary),
                "size_bytes": binary.stat().st_size,
                "codesign": fake_codesign_metadata(binary),
            }
        )
    payload: dict[str, object] = {
        "schema": native.NATIVE_RUNTIME_MANIFEST_SCHEMA,
        "target": {
            "python_major_minor": "3.12",
            "soabi": "cpython-312-darwin",
            "arch": "arm64",
            "extension_suffix": native.NATIVE_EXTENSION_SUFFIX,
        },
        "source_build_manifest": {
            "filename": native.NATIVE_BUILD_MANIFEST_NAME,
            "sha256": native.sha256_file(build_manifest),
        },
        "modules": modules,
        "signing": {
            "mode": "developer_id",
            "team_identifier": TEST_TEAM_IDENTIFIER,
            "nested_modules_verified": True,
        },
    }
    manifest_path = write_test_native_manifest(root, payload)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def make_test_build_payload(
    root: Path,
) -> tuple[Path, Path, Path, dict[str, object]]:
    payload_dir = root / "payload"
    source_dir = root / "source" / "pipeline"
    project_root = root / "source"
    payload_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    modules: list[dict[str, object]] = []
    for index, name in enumerate(native.NATIVE_MODULES):
        source = source_dir / f"{name}.py"
        source.write_text(f"VALUE = {index}\n", encoding="utf-8")
        filename = f"{name}{native.NATIVE_EXTENSION_SUFFIX}"
        binary = payload_dir / filename
        binary.write_bytes(f"build-{index}-{name}".encode("utf-8"))
        modules.append(
            {
                "import_name": name,
                "binary_path": f"pipeline/{filename}",
                "binary_sha256": native.sha256_file(binary),
                "size_bytes": binary.stat().st_size,
                "source_set_sha256": native.sha256_file(source),
            }
        )
    manifest: dict[str, object] = {
        "schema": native.NATIVE_BUILD_MANIFEST_SCHEMA,
        "target": {
            "python_version": "3.12.9",
            "soabi": "cpython-312-darwin",
            "arch": "arm64",
            "extension_suffix": native.NATIVE_EXTENSION_SUFFIX,
        },
        "native_scope": {"modules": modules},
        "source": {"build_inputs": []},
        "blocking_reasons": [],
    }
    manifest["manifest_payload_sha256"] = native.canonical_payload_sha256(
        manifest
    )
    (payload_dir / native_manager.SOURCE_BUILD_MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload_dir, source_dir, project_root, manifest


class NativePipelineRuntimeTests(unittest.TestCase):

    def setUp(self) -> None:
        codesign = mock.patch.object(
            native,
            "_codesign_metadata",
            side_effect=fake_codesign_metadata,
            create=True,
        )
        unsigned = mock.patch.object(
            native,
            "unsigned_macho_sha256",
            side_effect=fake_unsigned_macho_sha256,
            create=True,
        )
        codesign.start()
        unsigned.start()
        self.addCleanup(codesign.stop)
        self.addCleanup(unsigned.stop)

    def test_valid_runtime_payload_has_exact_native_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "pipeline"
            make_test_native_payload(root)

            manifest = native.validate_native_runtime_payload(root)

            self.assertEqual(
                tuple(record["import_name"] for record in manifest["modules"]),
                native.NATIVE_MODULES,
            )

    def test_manifest_payload_hash_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "pipeline"
            make_test_native_payload(root)
            manifest_path = root / native.NATIVE_RUNTIME_MANIFEST_NAME
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["target"]["arch"] = "x86_64"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                native.NativePipelineValidationError,
                "manifest payload hash mismatch",
            ):
                native.validate_native_runtime_payload(root)

    def test_expected_manifest_hash_rejects_a_different_self_consistent_payload(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "pipeline"
            first = make_test_native_payload(root, marker="first")
            expected = str(first["manifest_payload_sha256"])
            second = make_test_native_payload(root, marker="second")
            self.assertNotEqual(expected, second["manifest_payload_sha256"])

            with self.assertRaises(native.NativePipelineValidationError):
                native.validate_native_runtime_payload(
                    root,
                    expected_manifest_payload_sha256=expected,
                )

    def test_runtime_manifest_signature_evidence_is_reverified(self) -> None:
        mutations = (
            ("manifest_mode", "ad_hoc"),
            ("manifest_team", "OTHERTEAM"),
            ("actual_mode", "ad_hoc"),
            ("actual_team", "OTHERTEAM"),
            ("actual_cdhash", "0" * 40),
            ("actual_hardened", False),
        )
        for mutation, value in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                root = Path(td) / "pipeline"
                payload = make_test_native_payload(root)
                if mutation.startswith("manifest_"):
                    key = {
                        "manifest_mode": "mode",
                        "manifest_team": "team_identifier",
                    }[mutation]
                    payload["signing"][key] = value
                    write_test_native_manifest(root, payload)
                else:
                    actual = fake_codesign_metadata(
                        root
                        / (
                            native.NATIVE_MODULES[0]
                            + native.NATIVE_EXTENSION_SUFFIX
                        )
                    )
                    if mutation == "actual_mode":
                        actual["mode"] = value
                    elif mutation == "actual_team":
                        actual["team_identifier"] = value
                    elif mutation == "actual_cdhash":
                        actual["cdhash"] = value
                    else:
                        actual["hardened_runtime"] = value
                    with mock.patch.object(
                        native,
                        "_codesign_metadata",
                        return_value=actual,
                    ):
                        with self.assertRaises(
                            native.NativePipelineValidationError
                        ):
                            native.validate_native_runtime_payload(root)
                    continue

                with self.assertRaises(native.NativePipelineValidationError):
                    native.validate_native_runtime_payload(root)

    def test_runtime_manifest_unsigned_macho_lineage_is_reverified(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "pipeline"
            make_test_native_payload(root)

            with (
                mock.patch.object(
                    native,
                    "unsigned_macho_sha256",
                    return_value="0" * 64,
                ),
                self.assertRaises(native.NativePipelineValidationError),
            ):
                native.validate_native_runtime_payload(root)

    def test_missing_or_tampered_native_binary_is_rejected(self) -> None:
        for mutation, expected in (
            ("missing", "missing or is a symlink"),
            ("tampered", "size mismatch|hash mismatch"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                root = Path(td) / "pipeline"
                make_test_native_payload(root)
                binary = root / (
                    native.NATIVE_MODULES[0] + native.NATIVE_EXTENSION_SUFFIX
                )
                if mutation == "missing":
                    binary.unlink()
                else:
                    binary.write_bytes(binary.read_bytes() + b"tampered")

                with self.assertRaisesRegex(
                    native.NativePipelineValidationError,
                    expected,
                ):
                    native.validate_native_runtime_payload(root)

    def test_extra_native_binary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "pipeline"
            make_test_native_payload(root)
            (root / f"unexpected{native.NATIVE_EXTENSION_SUFFIX}").write_bytes(
                b"extra"
            )

            with self.assertRaisesRegex(
                native.NativePipelineValidationError,
                "exact inventory mismatch",
            ):
                native.validate_native_runtime_payload(root)

    def test_matching_source_or_bytecode_fallback_is_rejected(self) -> None:
        name = native.NATIVE_MODULES[0]
        fallback_paths = (
            Path(f"{name}.py"),
            Path(f"{name}.pyc"),
            Path("__pycache__") / f"{name}.cpython-312.pyc",
        )
        for relative in fallback_paths:
            with self.subTest(path=str(relative)), tempfile.TemporaryDirectory() as td:
                root = Path(td) / "pipeline"
                make_test_native_payload(root)
                fallback = root / relative
                fallback.parent.mkdir(parents=True, exist_ok=True)
                fallback.write_bytes(b"source fallback")

                with self.assertRaisesRegex(
                    native.NativePipelineValidationError,
                    "fallback is forbidden",
                ):
                    native.validate_native_runtime_payload(root)

    def test_formal_app_requires_native_but_development_allows_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "pipeline"
            root.mkdir()
            for name in native.NATIVE_MODULES:
                (root / f"{name}.py").write_text("VALUE = 1\n", encoding="utf-8")

            formal = native.inspect_native_pipeline(root, required=True)
            development = native.inspect_native_pipeline(root, required=False)

            self.assertFalse(formal["available"])
            self.assertEqual(formal["mode"], "native_missing")
            self.assertTrue(development["available"])
            self.assertEqual(development["mode"], "source")

    def test_invalid_manifest_never_downgrades_to_development_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "pipeline"
            root.mkdir()
            for name in native.NATIVE_MODULES:
                (root / f"{name}.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / native.NATIVE_RUNTIME_MANIFEST_NAME).write_text(
                "{}\n",
                encoding="utf-8",
            )

            capability = native.inspect_native_pipeline(root, required=False)

            self.assertFalse(capability["available"])
            self.assertEqual(capability["mode"], "native_invalid")

    def test_stage_runtime_payload_validates_before_copy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            destination = root / "runtime"
            make_test_native_payload(source)
            binary = source / (
                native.NATIVE_MODULES[0] + native.NATIVE_EXTENSION_SUFFIX
            )
            binary.write_bytes(binary.read_bytes() + b"tampered")

            with self.assertRaises(native.NativePipelineValidationError):
                native.stage_native_runtime_payload(source, destination)

            self.assertFalse(destination.exists())

    def test_stage_runtime_payload_validates_after_copy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            destination = root / "runtime"
            make_test_native_payload(source)
            original_copy2 = native.shutil.copy2
            first_binary = (
                native.NATIVE_MODULES[0] + native.NATIVE_EXTENSION_SUFFIX
            )

            def corrupt_after_copy(src, dst, *args, **kwargs):
                result = original_copy2(src, dst, *args, **kwargs)
                if Path(src).name == first_binary:
                    Path(dst).write_bytes(Path(dst).read_bytes() + b"tampered")
                return result

            with (
                mock.patch.object(
                    native.shutil,
                    "copy2",
                    side_effect=corrupt_after_copy,
                ),
                self.assertRaisesRegex(
                    native.NativePipelineValidationError,
                    "size mismatch|hash mismatch",
                ),
            ):
                native.stage_native_runtime_payload(source, destination)

    def test_stage_runtime_payload_rejects_a_coherent_source_swap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            replacement = root / "replacement"
            destination = root / "runtime"
            original = make_test_native_payload(source, marker="original")
            make_test_native_payload(replacement, marker="replacement")
            original_copy2 = native.shutil.copy2
            swapped = False

            def swap_payload_before_copy(src, dst, *args, **kwargs):
                nonlocal swapped
                if not swapped:
                    swapped = True
                    for item in replacement.iterdir():
                        original_copy2(item, source / item.name)
                return original_copy2(src, dst, *args, **kwargs)

            with (
                mock.patch.object(
                    native.shutil,
                    "copy2",
                    side_effect=swap_payload_before_copy,
                ),
                self.assertRaises(native.NativePipelineValidationError),
            ):
                native.stage_native_runtime_payload(source, destination)

            if destination.exists():
                with self.assertRaises(native.NativePipelineValidationError):
                    native.validate_native_runtime_payload(
                        destination,
                        expected_manifest_payload_sha256=str(
                            original["manifest_payload_sha256"]
                        ),
                    )

    def test_stage_runtime_payload_copies_only_verified_native_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            destination = root / "runtime"
            make_test_native_payload(source)

            native.stage_native_runtime_payload(source, destination)

            native.validate_native_runtime_payload(destination)
            self.assertEqual(
                {path.name for path in destination.glob("*.so")},
                {
                    f"{name}{native.NATIVE_EXTENSION_SUFFIX}"
                    for name in native.NATIVE_MODULES
                },
            )
            for name in native.NATIVE_MODULES:
                self.assertFalse((destination / f"{name}.py").exists())
                self.assertFalse((destination / f"{name}.pyc").exists())


class NativePipelineBundleManagerTests(unittest.TestCase):

    def _embed_with_receipt(
        self,
        root: Path,
    ) -> tuple[Path, dict[str, object]]:
        payload, source, project, _manifest = make_test_build_payload(root)
        destination = root / "pipeline"
        with mock.patch.object(
            native_manager,
            "unsigned_macho_sha256",
            side_effect=fake_unsigned_macho_sha256,
        ):
            result = native_manager.embed_build_payload(
                payload,
                source_dir=source,
                destination_dir=destination,
                project_root=project,
            )
        return destination, result


    def test_embed_rejects_a_symlink_destination_root(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        ) as td:
            root = Path(td)
            payload, source, project, _manifest = make_test_build_payload(root)
            outside = root / "outside"
            outside.mkdir()
            sentinel = outside / f"{native.NATIVE_MODULES[0]}.py"
            sentinel.write_text("do not delete\n", encoding="utf-8")
            destination = root / "pipeline"
            destination.symlink_to(outside, target_is_directory=True)

            with self.assertRaises(native.NativePipelineValidationError):
                native_manager.embed_build_payload(
                    payload,
                    source_dir=source,
                    destination_dir=destination,
                    project_root=project,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not delete\n")

    def test_embed_rejects_a_symlink_pycache(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        ) as td:
            root = Path(td)
            payload, source, project, _manifest = make_test_build_payload(root)
            destination = root / "pipeline"
            destination.mkdir()
            outside = root / "outside-cache"
            outside.mkdir()
            sentinel = outside / (
                f"{native.NATIVE_MODULES[0]}.cpython-312.pyc"
            )
            sentinel.write_bytes(b"do not delete")
            (destination / "__pycache__").symlink_to(
                outside,
                target_is_directory=True,
            )

            with self.assertRaises(native.NativePipelineValidationError):
                native_manager.embed_build_payload(
                    payload,
                    source_dir=source,
                    destination_dir=destination,
                    project_root=project,
                )

            self.assertEqual(sentinel.read_bytes(), b"do not delete")

    def test_embed_creates_a_hashed_seal_receipt(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        ) as td:
            destination, result = self._embed_with_receipt(Path(td))
            receipt_path = destination / native_manager.NATIVE_EMBED_RECEIPT_NAME

            self.assertTrue(receipt_path.is_file())
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(
                result["seal_receipt_sha256"],
                receipt["manifest_payload_sha256"],
            )
            unsigned = dict(receipt)
            unsigned.pop("manifest_payload_sha256")
            self.assertEqual(
                receipt["manifest_payload_sha256"],
                native.canonical_payload_sha256(unsigned),
            )

    def test_seal_rejects_an_unexpected_receipt_identity(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        ) as td:
            destination, _result = self._embed_with_receipt(Path(td))

            with self.assertRaises(native.NativePipelineValidationError):
                native_manager.seal_runtime_payload(
                    destination,
                    target_python=Path("/unused/python3.12"),
                    app_bundle_id="com.example.Starun",
                    app_version="1.0",
                    signing_mode="developer_id",
                    expected_receipt_sha256="0" * 64,
                    expected_team_identifier=TEST_TEAM_IDENTIFIER,
                )

    def test_seal_rejects_build_manifest_changed_after_embed(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        ) as td:
            destination, result = self._embed_with_receipt(Path(td))
            build_manifest = destination / native.NATIVE_BUILD_MANIFEST_NAME
            build_manifest.write_text(
                build_manifest.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(native.NativePipelineValidationError):
                native_manager.seal_runtime_payload(
                    destination,
                    target_python=Path("/unused/python3.12"),
                    app_bundle_id="com.example.Starun",
                    app_version="1.0",
                    signing_mode="developer_id",
                    expected_receipt_sha256=str(result["seal_receipt_sha256"]),
                    expected_team_identifier=TEST_TEAM_IDENTIFIER,
                )

    def test_seal_rejects_unsigned_macho_lineage_changed_after_embed(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        ) as td:
            destination, result = self._embed_with_receipt(Path(td))
            binary = destination / (
                native.NATIVE_MODULES[0] + native.NATIVE_EXTENSION_SUFFIX
            )
            binary.write_bytes(binary.read_bytes() + b"replacement")

            with (
                mock.patch.object(
                    native_manager,
                    "unsigned_macho_sha256",
                    side_effect=fake_unsigned_macho_sha256,
                ),
                mock.patch.object(
                    native_manager,
                    "_target_python_metadata",
                    return_value={
                        "python_version": "3.12.9",
                        "python_major_minor": "3.12",
                        "arch": "arm64",
                        "soabi": "cpython-312-darwin",
                        "extension_suffix": native.NATIVE_EXTENSION_SUFFIX,
                    },
                ),
                self.assertRaises(native.NativePipelineValidationError),
            ):
                native_manager.seal_runtime_payload(
                    destination,
                    target_python=Path("/unused/python3.12"),
                    app_bundle_id="com.example.Starun",
                    app_version="1.0",
                    signing_mode="developer_id",
                    expected_receipt_sha256=str(result["seal_receipt_sha256"]),
                    expected_team_identifier=TEST_TEAM_IDENTIFIER,
                )

    def test_seal_does_not_follow_a_runtime_manifest_symlink(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        ) as td:
            root = Path(td)
            destination, result = self._embed_with_receipt(root)
            outside = root / "outside-runtime-manifest.json"
            outside.write_text("do not overwrite\n", encoding="utf-8")
            runtime_manifest = destination / native.NATIVE_RUNTIME_MANIFEST_NAME
            runtime_manifest.symlink_to(outside)

            with (
                mock.patch.object(
                    native_manager,
                    "unsigned_macho_sha256",
                    side_effect=fake_unsigned_macho_sha256,
                ),
                mock.patch.object(
                    native_manager,
                    "_target_python_metadata",
                    return_value={
                        "python_version": "3.12.9",
                        "python_major_minor": "3.12",
                        "arch": "arm64",
                        "soabi": "cpython-312-darwin",
                        "extension_suffix": native.NATIVE_EXTENSION_SUFFIX,
                    },
                ),
                mock.patch.object(
                    native_manager,
                    "_codesign_metadata",
                    side_effect=lambda path, **_kwargs: fake_codesign_metadata(path),
                ),
                mock.patch.object(
                    native_manager,
                    "validate_native_runtime_payload",
                    side_effect=lambda pipeline_root: json.loads(
                        (
                            Path(pipeline_root)
                            / native.NATIVE_RUNTIME_MANIFEST_NAME
                        ).read_text(encoding="utf-8")
                    ),
                ),
            ):
                native_manager.seal_runtime_payload(
                    destination,
                    target_python=Path("/unused/python3.12"),
                    app_bundle_id="com.example.Starun",
                    app_version="1.0",
                    signing_mode="developer_id",
                    expected_receipt_sha256=str(result["seal_receipt_sha256"]),
                    expected_team_identifier=TEST_TEAM_IDENTIFIER,
                )

            self.assertEqual(outside.read_text(encoding="utf-8"), "do not overwrite\n")
            self.assertTrue(runtime_manifest.is_file())
            self.assertFalse(runtime_manifest.is_symlink())


if __name__ == "__main__":
    unittest.main()
