#!/usr/bin/env python3
"""Apply Starun's read-only/offline Zenith integration to SyQon Starless."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SYQON_RELATIVE_CANDIDATES = (
    Path("vendor/siril-scripts/SyQon/Starless.py"),
    Path("vendor/siril-scripts/siril-scripts/SyQon/Starless.py"),
    Path("SyQon/Starless.py"),
    Path("Starless.py"),
)
PATCH_SENTINEL = "STARUN_OFFLINE_MODEL_PATCH = True"
UPSTREAM_COMMIT = "4cc9e204f9ddfd6d03cc4283aac76c82d4d19167"
UPSTREAM_STARLESS_SHA256 = (
    "d36818f24a6927b245ab66fc7c00eaaa3b330a47406b61f0e9beb0764e06ab11"
)

CONSTANTS_OLD = """\
BASE_URL           = "https://siril.syqon.it"
SCRIPT_VERSION     = "3.0"
ZENITH_MODEL_PATH  = None   # set to engine_dir / "zenith.pt" in main()
AXIOM3_MODEL_PATH  = None   # set to engine_dir / "axiom3.pt" in main()
"""

CONSTANTS_NEW = CONSTANTS_OLD + """\
ENV_SYQON_MODEL_DIR_KEY = "STARUN_SYQON_MODEL_DIR"
ENV_NETWORK_MODE_KEY    = "STARUN_NETWORK_MODE"
ENV_TRUE_VALUES         = frozenset({"1", "true", "yes", "on"})
STARUN_OFFLINE_MODEL_PATCH = True
"""

DOWNLOAD_START_OLD = """\
def download_file(url, destination, description, silent=False):
    if not silent:
"""

DOWNLOAD_START_NEW = """\
def syqon_network_downloads_allowed():
    \"\"\"Project runtime requires an explicit opt-in before any model download.\"\"\"
    return os.getenv(ENV_NETWORK_MODE_KEY, "0").strip().lower() in ENV_TRUE_VALUES


def resolve_zenith_model_dir(runtime_engine_dir: Path):
    \"\"\"Prefer the project-owned, read-only Zenith bundle over Siril runtime data.\"\"\"
    configured = os.getenv(ENV_SYQON_MODEL_DIR_KEY, "").strip()
    if configured:
        configured_dir = Path(configured).expanduser().resolve()
        if configured_dir.is_file() and configured_dir.name == "zenith.pt":
            configured_dir = configured_dir.parent
        if not (configured_dir / "zenith.pt").is_file():
            raise FileNotFoundError(
                f"{ENV_SYQON_MODEL_DIR_KEY} does not contain zenith.pt: "
                f"{configured_dir}"
            )
        return configured_dir, ENV_SYQON_MODEL_DIR_KEY

    try:
        bundled_dir = Path(__file__).resolve().parents[3] / "syqon_starless"
    except IndexError:
        bundled_dir = None
    if bundled_dir is not None and (bundled_dir / "zenith.pt").is_file():
        return bundled_dir.resolve(), "project bundle"

    return runtime_engine_dir, "Siril runtime"


def download_file(url, destination, description, silent=False):
    if not syqon_network_downloads_allowed():
        if not silent:
            print(
                f"Blocked download of {description}: "
                f"{ENV_NETWORK_MODE_KEY} is not enabled",
                file=sys.stderr,
            )
        return False
    if not silent:
"""

UPDATE_CHECK_OLD = """\
def should_check_for_updates(engine_dir, force_update_check=False):
    if not has_network_connectivity():
"""

UPDATE_CHECK_NEW = """\
def should_check_for_updates(engine_dir, force_update_check=False):
    if not syqon_network_downloads_allowed():
        print(f"Offline ({ENV_NETWORK_MODE_KEY}=0), skipping update check")
        return False
    if not has_network_connectivity():
"""

VERIFY_OLD = """\
def verify_shasum(file_path, shasum_file):
    with open(shasum_file, "r") as f:
        expected_hash = f.read().strip().split()[0]
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest() == expected_hash
"""

VERIFY_NEW = VERIFY_OLD + """\


def validate_local_zenith_model(model_dir: Path):
    \"\"\"Validate a local Zenith model without writing beside read-only assets.\"\"\"
    model_file = model_dir / "zenith.pt"
    shasum_file = model_dir / "zenith.pt.sha256"
    if not model_file.is_file():
        return False, f"Zenith model not found: {model_file}"
    if not shasum_file.is_file():
        return False, f"Zenith checksum not found: {shasum_file}"
    try:
        if not verify_shasum(model_file, shasum_file):
            return False, f"Zenith SHA256 verification failed: {model_file}"
    except (OSError, ValueError, IndexError) as error:
        return False, f"Could not verify Zenith model: {error}"
    return True, "SHA256 verification passed"
"""

DOWNLOAD_BRANCH_OLD = """\
    if should_download:
        url        = f"{BASE_URL}/{file_name}"
"""

DOWNLOAD_BRANCH_NEW = """\
    if should_download:
        if not syqon_network_downloads_allowed():
            print(
                "Error: Zenith model download blocked because "
                f"{ENV_NETWORK_MODE_KEY} is not enabled.",
                file=sys.stderr,
            )
            sys.exit(1)
        url        = f"{BASE_URL}/{file_name}"
"""

MAIN_MODEL_OLD = """\
    engine_dir = Path(user_datadir) / "syqon_starless"
    engine_dir.mkdir(parents=True, exist_ok=True)

    # Set global model paths
    global ZENITH_MODEL_PATH, AXIOM3_MODEL_PATH
    ZENITH_MODEL_PATH  = engine_dir / "zenith.pt"
__SYQON_INDENTED_BLANK__
    update_axiom3_path(engine_dir)
    if AXIOM3_MODEL_PATH.exists():
        print(f"Using Axiom V3 model: {AXIOM3_MODEL_PATH}")

    # Setup / download Zenith model
    should_check = should_check_for_updates(engine_dir, args.force_update_check)
    setup_model_torch(engine_dir, siril, should_check)
""".replace("__SYQON_INDENTED_BLANK__", "    ")

MAIN_MODEL_NEW = """\
    runtime_engine_dir = Path(user_datadir) / "syqon_starless"
    try:
        engine_dir, model_source = resolve_zenith_model_dir(runtime_engine_dir)
    except (OSError, ValueError) as e:
        print(f"Error: Could not resolve local Zenith model: {e}", file=sys.stderr)
        sys.exit(1)

    network_downloads_allowed = syqon_network_downloads_allowed()
    if model_source == "Siril runtime" and network_downloads_allowed:
        engine_dir.mkdir(parents=True, exist_ok=True)

    # Set global model paths
    global ZENITH_MODEL_PATH, AXIOM3_MODEL_PATH
    ZENITH_MODEL_PATH  = engine_dir / "zenith.pt"

    update_axiom3_path(engine_dir)
    if AXIOM3_MODEL_PATH.exists():
        print(f"Using Axiom V3 model: {AXIOM3_MODEL_PATH}")

    # Bundled/offline models are immutable inputs: verify and never update them.
    if model_source != "Siril runtime" or not network_downloads_allowed:
        model_valid, model_message = validate_local_zenith_model(engine_dir)
        if not model_valid:
            print(f"Error: {model_message}", file=sys.stderr)
            print(
                "SyQon will not download a replacement while using the offline "
                "model path.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Using local Zenith model from {model_source}: {ZENITH_MODEL_PATH}")
        print(model_message)
        if args.force_update_check:
            print("Ignoring --force_update_check for the local offline model")
    else:
        should_check = should_check_for_updates(engine_dir, args.force_update_check)
        setup_model_torch(engine_dir, siril, should_check)
"""

SAFE_HEADER_OLD = '''\
def _make_safe_header(raw_header: str, new_filter: str) -> str:
    if not raw_header:
        return ""
    try:
        lines = raw_header.split("\\n")
        lines = [l for l in lines if not (l.strip().startswith("END") or
                                           l.strip().startswith("CONTINUE") or
                                           l.strip().endswith("&'"))]
        while lines and not lines[-1].strip():
            lines.pop()
        header_dict = fits.Header.fromstring("\\n".join(lines), sep="\\n")
        header_dict["FILTER"] = new_filter
        header_dict.remove("BG-PTS", ignore_missing=True)
        header_dict.verify("silentfix")
        return header_dict.tostring(sep="\\n")
    except Exception as e:
        print(f"Warning: safe FITS header creation failed for filter '{new_filter}': {e}. Using raw header.")
        return raw_header
'''

SAFE_HEADER_NEW = '''\
def _make_safe_header(raw_header: str, new_filter: str) -> str:
    try:
        lines = raw_header.split("\\n") if raw_header else []
        lines = [l for l in lines if not (l.strip().startswith("END") or
                                           l.strip().startswith("CONTINUE") or
                                           l.strip().endswith("&'"))]
        while lines and not lines[-1].strip():
            lines.pop()
        header_dict = (
            fits.Header.fromstring("\\n".join(lines), sep="\\n")
            if lines
            else fits.Header()
        )
        header_dict["FILTER"] = new_filter
        # Output pixels always use canonical float32 0..1. Never retain source
        # integer scaling cards: they would reinterpret the new samples on load.
        for keyword in (
            "BITPIX",
            "BSCALE",
            "BZERO",
            "BLANK",
            "DATAMIN",
            "DATAMAX",
            "BG-PTS",
        ):
            header_dict.remove(keyword, ignore_missing=True)
        return header_dict.tostring(sep="\\n")
    except Exception as e:
        print(f"Warning: safe FITS header creation failed for filter '{new_filter}': {e}. Using a minimal header.")
        header_dict = fits.Header()
        header_dict["FILTER"] = new_filter
        return header_dict.tostring(sep="\\n")
'''

RESTORE_DTYPE_OLD = '''\
def restore_image_dtype(data: np.ndarray, original_dtype: np.dtype, scale_factor: float) -> np.ndarray:
    if original_dtype == np.float32:
        return data.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        scaled  = data * scale_factor
        clipped = np.clip(scaled, 0, scale_factor)
        return clipped.astype(original_dtype)
    return data.astype(original_dtype)
'''

RESTORE_DTYPE_NEW = '''\
def restore_image_dtype(data: np.ndarray, original_dtype: np.dtype, scale_factor: float) -> np.ndarray:
    """Return the canonical Siril exchange domain: finite float32 in 0..1.

    ``original_dtype`` and ``scale_factor`` are intentionally retained in the
    signature for upstream call-site compatibility. Re-quantizing model output
    to the source integer dtype and then carrying its FITS scale cards caused a
    second, ambiguous numerical conversion at the Siril boundary.
    """
    _ = (original_dtype, scale_factor)
    canonical = np.asarray(data, dtype=np.float32)
    if canonical.size == 0:
        raise ValueError("SyQon output is empty")
    if not np.all(np.isfinite(canonical)):
        raise ValueError("SyQon output contains NaN or Inf")
    tolerance = 1e-4
    output_min = float(np.min(canonical))
    output_max = float(np.max(canonical))
    if output_min < -tolerance or output_max > 1.0 + tolerance:
        raise ValueError(
            "SyQon output violates canonical 0..1 domain: "
            f"min={output_min:.9g}, max={output_max:.9g}"
        )
    return np.clip(canonical, 0.0, 1.0).astype(np.float32, copy=False)
'''

GUI_SEQUENCE_HEADER_OLD = '''\
                    lines    = frame_obj.header.split("\\n")
                    lines    = [l for l in lines if not (l.strip().startswith("END") or
                                                          l.strip().startswith("CONTINUE") or
                                                          l.strip().endswith("&'"))]
                    while lines and not lines[-1].strip():
                        lines.pop()
                    header_dict = fits.Header.fromstring("\\n".join(lines), sep="\\n")
                    header_dict["FILTER"] = "starless"
                    header_str = header_dict.tostring(sep="\\n")
                    base = os.path.splitext(os.path.basename(filename))[0]
                    output_dir = gui.get_output_dir(cwd)
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    new_filename = os.path.join(output_dir, ("starless_" + base) if not base.startswith("starless_") else base)
                    siril.save_image_file(starless_restored, header_str, new_filename)
                    if mask_restored is not None:
                        header_dict["FILTER"] = "starmask"
                        header_str   = header_dict.tostring(sep="\\n")
'''

GUI_SEQUENCE_HEADER_NEW = '''\
                    header_str = _make_safe_header(frame_obj.header, "starless")
                    base = os.path.splitext(os.path.basename(filename))[0]
                    output_dir = gui.get_output_dir(cwd)
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    new_filename = os.path.join(output_dir, ("starless_" + base) if not base.startswith("starless_") else base)
                    siril.save_image_file(starless_restored, header_str, new_filename)
                    if mask_restored is not None:
                        header_str = _make_safe_header(frame_obj.header, "starmask")
'''

CLI_SINGLE_HEADER_OLD = '''\
                    lines = input_image.header.split("\\n")
                    lines = [l for l in lines if not l.strip().startswith("END")]
                    while lines and not lines[-1].strip():
                        lines.pop()
                    header_dict = fits.Header.fromstring("\\n".join(lines), sep="\\n")
                    header_dict["FILTER"] = "starless"
                    hdu = fits.PrimaryHDU(header=header_dict)
                    hdu.header.remove("BG-PTS", ignore_missing=True)
                    hdu.verify("silentfix")
                    header_str = hdu.header.tostring(sep="\\n")
                    filename   = siril.get_image_filename()
'''

CLI_SINGLE_HEADER_NEW = '''\
                    header_str = _make_safe_header(input_image.header, "starless")
                    filename   = siril.get_image_filename()
'''

CLI_SINGLE_MASK_HEADER_OLD = '''\
                    if mask_restored is not None:
                        header_dict["FILTER"] = "starmask"
                        hdu = fits.PrimaryHDU(header=header_dict)
                        hdu.header.remove("BG-PTS", ignore_missing=True)
                        hdu.verify("silentfix")
                        header_str    = hdu.header.tostring(sep="\\n")
                        base_starmask = base if base.startswith("starmask_") else "starmask_" + base
'''

CLI_SINGLE_MASK_HEADER_NEW = '''\
                    if mask_restored is not None:
                        header_str = _make_safe_header(input_image.header, "starmask")
                        base_starmask = base if base.startswith("starmask_") else "starmask_" + base
'''

CLI_SEQUENCE_HEADER_OLD = '''\
                    lines = frame.header.split("\\n")
                    lines = [l for l in lines if not l.strip().startswith("END")]
                    while lines and not lines[-1].strip():
                        lines.pop()
                    header_dict = fits.Header.fromstring("\\n".join(lines), sep="\\n")
                    header_dict["FILTER"] = "starless"
                    header_str   = header_dict.tostring(sep="\\n")
                    base         = os.path.splitext(os.path.basename(filename))[0]
                    new_filename = os.path.join(cwd, "starless_" + base)
                    siril.save_image_file(starless_restored, header_str, new_filename)
                    print(f"Saved starless: {new_filename}")

                    if mask_restored is not None:
                        header_dict["FILTER"] = "starmask"
                        header_str   = header_dict.tostring(sep="\\n")
'''

CLI_SEQUENCE_HEADER_NEW = '''\
                    header_str = _make_safe_header(frame.header, "starless")
                    base         = os.path.splitext(os.path.basename(filename))[0]
                    new_filename = os.path.join(cwd, "starless_" + base)
                    siril.save_image_file(starless_restored, header_str, new_filename)
                    print(f"Saved starless: {new_filename}")

                    if mask_restored is not None:
                        header_str = _make_safe_header(frame.header, "starmask")
'''

WEIGHT_CACHE_OLD = '''\
def _edge_key(pos: int, size: int, tile: int) -> str:
    if pos == 0:
        return "top"
    if pos + tile >= size:
        return "bot"
    return "mid"


def _build_weight_cache(tile_size: int, overlap: int, device: torch.device, dtype: torch.dtype) -> dict:
    def axis_weights(size, overlap, device, dtype):
        base = _weight_1d(size, overlap, device, dtype)
        top  = base.clone()
        bot  = base.clone()
        if overlap > 0:
            top[:overlap]  = 1.0
            bot[-overlap:] = 1.0
        return top, base, bot

    wy_top, wy_mid, wy_bot    = axis_weights(tile_size, overlap, device, dtype)
    wx_left, wx_mid, wx_right = axis_weights(tile_size, overlap, device, dtype)

    cache = {}
    for yk, wy in (("top", wy_top), ("mid", wy_mid), ("bot", wy_bot)):
        for xk, wx in (("left", wx_left), ("mid", wx_mid), ("right", wx_right)):
            cache[(yk, xk)] = (wy[:, None] * wx[None, :]).unsqueeze(0).unsqueeze(0)
    return cache
'''

WEIGHT_CACHE_NEW = '''\
def _edge_key(pos: int, size: int, tile: int) -> str:
    if pos == 0 and pos + tile >= size:
        return "solo"
    if pos == 0:
        return "top"
    if pos + tile >= size:
        return "bot"
    return "mid"


def _build_weight_cache(tile_size: int, overlap: int, device: torch.device, dtype: torch.dtype) -> dict:
    def axis_weights(size, overlap, device, dtype):
        base = _weight_1d(size, overlap, device, dtype)
        top  = base.clone()
        bot  = base.clone()
        solo = torch.ones(size, device=device, dtype=dtype)
        if overlap > 0:
            top[:overlap]  = 1.0
            bot[-overlap:] = 1.0
        return top, base, bot, solo

    wy_top, wy_mid, wy_bot, wy_solo = axis_weights(
        tile_size, overlap, device, dtype
    )
    wx_left, wx_mid, wx_right, wx_solo = axis_weights(
        tile_size, overlap, device, dtype
    )

    cache = {}
    for yk, wy in (
        ("top", wy_top),
        ("mid", wy_mid),
        ("bot", wy_bot),
        ("solo", wy_solo),
    ):
        for xk, wx in (
            ("left", wx_left),
            ("mid", wx_mid),
            ("right", wx_right),
            ("solo", wx_solo),
        ):
            cache[(yk, xk)] = (wy[:, None] * wx[None, :]).unsqueeze(0).unsqueeze(0)
    return cache
'''

TILE_INFERENCE_OLD = '''\
def tile_inference_torch(
    model: nn.Module,
    x: torch.Tensor,
    tile_size: int,
    overlap: int,
    device: torch.device,
    amp_enabled: bool,
    progress_callback: Optional[Callable[[int], None]] = None,
    progress_start: int = 10,
    progress_range: int = 75,
    tile_callback: Optional[Callable[[int, int, int, int, np.ndarray], None]] = None,
    tile_transform: Optional[Callable[[torch.Tensor, torch.Tensor], np.ndarray]] = None,
    check_cancel: Optional[Callable[[], bool]] = None,
) -> torch.Tensor:
    _, _, h, w = x.shape

    if tile_size <= 0 or tile_size >= min(h, w):
        if progress_callback:
            progress_callback(progress_start + progress_range // 2)
        with torch.no_grad(), amp.autocast(device_type=device.type, enabled=amp_enabled):
            result = model(x)
        if progress_callback:
            progress_callback(progress_start + progress_range)
        return result

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    ys = _tile_positions(h, tile_size, stride)
    xs = _tile_positions(w, tile_size, stride)
    total_tiles = len(ys) * len(xs)

    out    = torch.zeros_like(x, device=device)
    weight = torch.zeros((1, 1, h, w), device=device, dtype=x.dtype)
    cache  = _build_weight_cache(tile_size, overlap, device, x.dtype)
    xk_map = {"top": "left", "mid": "mid", "bot": "right"}

    tile_count = 0
    with torch.no_grad():
        for y in reversed(ys):
            yk = _edge_key(y, h, tile_size)
            for x0 in xs:
                if check_cancel is not None and check_cancel():
                    raise InterruptedError("Cancelled")
                xk  = xk_map[_edge_key(x0, w, tile_size)]
                w2  = cache[(yk, xk)]
                tile = x[:, :, y : y + tile_size, x0 : x0 + tile_size]
                try:
                    with amp.autocast(device_type=device.type, enabled=amp_enabled):
                        pred = model(tile)
                except Exception:
                    pred = model(tile)

                out   [:, :, y : y + tile_size, x0 : x0 + tile_size] += pred * w2
                weight[:, :, y : y + tile_size, x0 : x0 + tile_size] += w2

                if tile_callback is not None:
                    if tile_transform is not None:
                        patch = tile_transform(tile, pred)
                    else:
                        patch = pred[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float32)
                    tile_callback(y, x0, pred.shape[2], pred.shape[3], patch)

                tile_count += 1
                if progress_callback:
                    pct = progress_start + int((tile_count / total_tiles) * progress_range)
                    progress_callback(pct)

    return out / weight.clamp(min=1e-8)
'''

TILE_INFERENCE_NEW = '''\
STARUN_LAST_INFERENCE_DIAGNOSTICS = {}


def _starun_pad_zenith_tensor(x: torch.Tensor, tile_size: int):
    """Pad Zenith input to depth-4 geometry without leaking padding to output."""
    _, _, original_h, original_w = x.shape
    tiled = tile_size > 0 and (original_h > tile_size or original_w > tile_size)
    minimum_h = tile_size if tiled and original_h < tile_size else original_h
    minimum_w = tile_size if tiled and original_w < tile_size else original_w
    padded_h = ((minimum_h + 15) // 16) * 16
    padded_w = ((minimum_w + 15) // 16) * 16
    pad_h = padded_h - original_h
    pad_w = padded_w - original_w
    pad_mode = "none"
    if pad_h or pad_w:
        can_reflect = (
            original_h > 1
            and original_w > 1
            and pad_h < original_h
            and pad_w < original_w
        )
        pad_mode = "reflect" if can_reflect else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
    return x, {
        "original_shape": [int(original_h), int(original_w)],
        "padded_shape": [int(x.shape[2]), int(x.shape[3])],
        "padding": {"left": 0, "right": int(pad_w), "top": 0, "bottom": int(pad_h)},
        "padding_mode": pad_mode,
        "alignment": 16,
    }


def tile_inference_torch(
    model: nn.Module,
    x: torch.Tensor,
    tile_size: int,
    overlap: int,
    device: torch.device,
    amp_enabled: bool,
    progress_callback: Optional[Callable[[int], None]] = None,
    progress_start: int = 10,
    progress_range: int = 75,
    tile_callback: Optional[Callable[[int, int, int, int, np.ndarray], None]] = None,
    tile_transform: Optional[Callable[[torch.Tensor, torch.Tensor], np.ndarray]] = None,
    check_cancel: Optional[Callable[[], bool]] = None,
) -> torch.Tensor:
    global STARUN_LAST_INFERENCE_DIAGNOSTICS
    original_h, original_w = int(x.shape[2]), int(x.shape[3])
    x, geometry = _starun_pad_zenith_tensor(x, tile_size)
    _, _, h, w = x.shape

    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        if progress_callback:
            progress_callback(progress_start + progress_range // 2)
        with torch.no_grad(), amp.autocast(device_type=device.type, enabled=amp_enabled):
            result = model(x)
        result = result[:, :, :original_h, :original_w]
        if result.shape[2:] != (original_h, original_w) or not torch.isfinite(result).all():
            raise ValueError("Zenith full-frame output failed shape/finite contract")
        STARUN_LAST_INFERENCE_DIAGNOSTICS = {
            **geometry,
            "mode": "full_frame",
            "tile_size": int(tile_size),
            "overlap": int(overlap),
            "stride": None,
            "grid": {"rows": 1, "columns": 1, "tiles": 1},
            "coverage_min": 1.0,
            "coverage_max": 1.0,
            "crop_shape": [original_h, original_w],
        }
        if progress_callback:
            progress_callback(progress_start + progress_range)
        return result

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    ys = _tile_positions(h, tile_size, stride)
    xs = _tile_positions(w, tile_size, stride)
    total_tiles = len(ys) * len(xs)

    out    = torch.zeros_like(x, device=device)
    weight = torch.zeros((1, 1, h, w), device=device, dtype=x.dtype)
    cache  = _build_weight_cache(tile_size, overlap, device, x.dtype)
    xk_map = {"top": "left", "mid": "mid", "bot": "right", "solo": "solo"}

    tile_count = 0
    with torch.no_grad():
        for y in reversed(ys):
            yk = _edge_key(y, h, tile_size)
            for x0 in xs:
                if check_cancel is not None and check_cancel():
                    raise InterruptedError("Cancelled")
                xk  = xk_map[_edge_key(x0, w, tile_size)]
                w2  = cache[(yk, xk)]
                tile = x[:, :, y : y + tile_size, x0 : x0 + tile_size]
                try:
                    with amp.autocast(device_type=device.type, enabled=amp_enabled):
                        pred = model(tile)
                except Exception:
                    pred = model(tile)
                if pred.shape != tile.shape:
                    raise ValueError(
                        f"Zenith tile shape mismatch: tile={tuple(tile.shape)}, pred={tuple(pred.shape)}"
                    )

                out   [:, :, y : y + tile_size, x0 : x0 + tile_size] += pred * w2
                weight[:, :, y : y + tile_size, x0 : x0 + tile_size] += w2

                if tile_callback is not None:
                    if tile_transform is not None:
                        patch = tile_transform(tile, pred)
                    else:
                        patch = pred[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float32)
                    tile_callback(y, x0, pred.shape[2], pred.shape[3], patch)

                tile_count += 1
                if progress_callback:
                    pct = progress_start + int((tile_count / total_tiles) * progress_range)
                    progress_callback(pct)

    coverage_min = float(weight.min().detach().cpu())
    coverage_max = float(weight.max().detach().cpu())
    if not np.isfinite(coverage_min) or coverage_min <= 0.0:
        raise ValueError(f"Zenith tile coverage gap: minimum weight={coverage_min}")
    result = out / weight
    result = result[:, :, :original_h, :original_w]
    if result.shape[2:] != (original_h, original_w) or not torch.isfinite(result).all():
        raise ValueError("Zenith tiled output failed shape/finite contract")
    STARUN_LAST_INFERENCE_DIAGNOSTICS = {
        **geometry,
        "mode": "tiled",
        "tile_size": int(tile_size),
        "overlap": int(overlap),
        "stride": int(stride),
        "grid": {
            "rows": len(ys),
            "columns": len(xs),
            "tiles": total_tiles,
            "y_positions": [int(value) for value in ys],
            "x_positions": [int(value) for value in xs],
        },
        "coverage_min": coverage_min,
        "coverage_max": coverage_max,
        "crop_shape": [original_h, original_w],
    }
    return result
'''

PROCESS_IMAGE_ARGS_OLD = '''\
    stretch_method: str = "statistical",
    mtf_target: float = 0.15,
    linked_stretch: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
'''

PROCESS_IMAGE_ARGS_NEW = '''\
    stretch_method: str = "statistical",
    mtf_target: float = 0.15,
    linked_stretch: bool = False,
    stat_bp_sigma: float = 5.0,
    no_black_clip: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
'''

STAT_STRETCH_CALL_OLD = '''\
        stat_params = compute_stat_stretch_params(img_for_stretch, target_median=mtf_target)
'''

STAT_STRETCH_CALL_NEW = '''\
        stat_params = compute_stat_stretch_params(
            img_for_stretch,
            target_median=mtf_target,
            bp_sigma=stat_bp_sigma,
            no_black_clip=no_black_clip,
        )
'''

PROCESS_DIAGNOSTICS_OLD = '''\
    )
    _emit(87)

    # ---- apply residual mode -------------------------------------------------
'''

PROCESS_DIAGNOSTICS_NEW = '''\
    )
    STARUN_LAST_INFERENCE_DIAGNOSTICS.update({
        "device": str(device),
        "device_type": str(device.type),
        "actual_amp": bool(amp_enabled),
        "model": str(model_type),
        "residual_mode": bool(residual_mode),
        "stretch": {
            "method": str(stretch_method),
            "target_median": float(mtf_target if stretch_method != "ihs" else ihs_target),
            "linked": bool(linked_stretch),
            "stat_bp_sigma": float(stat_bp_sigma),
            "no_black_clip": bool(no_black_clip),
        },
        "mask_method": str(mask_method),
    })
    _emit(87)

    # ---- apply residual mode -------------------------------------------------
'''

PROCESS_ASYNC_ARGS_OLD = '''\
        mask_method: str = "subtraction",
        use_amp: bool = True,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
    ) -> None:
'''

PROCESS_ASYNC_ARGS_NEW = '''\
        mask_method: str = "subtraction",
        use_amp: bool = True,
        stretch_method: str = "statistical",
        target_median: float = 0.15,
        linked_stretch: bool = False,
        stat_bp_sigma: float = 5.0,
        no_black_clip: bool = False,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
    ) -> None:
'''

PROCESS_ASYNC_FORWARD_OLD = '''\
                    model_path=model_path,
                    model_type=model_type,
                )
'''

PROCESS_ASYNC_FORWARD_NEW = '''\
                    model_path=model_path,
                    model_type=model_type,
                    stretch_method=stretch_method,
                    mtf_target=target_median,
                    ihs_target=target_median,
                    linked_stretch=linked_stretch,
                    stat_bp_sigma=stat_bp_sigma,
                    no_black_clip=no_black_clip,
                )
'''

FILE_IO_HELPERS_OLD = '''\
def _siril_quoted_path(path: str) -> str:
    """Quote a filesystem path for Siril command parsing."""
    return '"' + str(path).replace("\\\\", "\\\\\\\\").replace('"', '\\\\"') + '"'


# ============================================================================
# Main entry point
# ============================================================================
'''

FILE_IO_HELPERS_NEW = '''\
def _siril_quoted_path(path: str) -> str:
    """Quote a filesystem path for Siril command parsing."""
    return '"' + str(path).replace("\\\\", "\\\\\\\\").replace('"', '\\\\"') + '"'


def _load_fits_file_input(input_path: Path):
    """Read one FITS image without opening a Siril connection."""
    resolved = input_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"SyQon input FITS not found: {resolved}")
    with fits.open(resolved, memmap=False, do_not_scale_image_data=False) as hdul:
        if not hdul or hdul[0].data is None:
            raise ValueError(f"SyQon input FITS has no primary image: {resolved}")
        pixels = np.array(hdul[0].data, copy=True)
        header_text = hdul[0].header.tostring(sep="\\n")
    prepared, original_dtype, scale_factor = prepare_image_for_inference(pixels)
    return resolved, prepared, original_dtype, scale_factor, header_text


def _write_fits_file_output(
    output_path: Path,
    pixels: np.ndarray,
    raw_header: str,
    filter_name: str,
) -> Path:
    """Atomically write canonical float32 FITS output without Siril."""
    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    header_text = _make_safe_header(raw_header, filter_name)
    header = fits.Header.fromstring(header_text, sep="\\n")
    temp_path = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    try:
        fits.PrimaryHDU(
            data=np.asarray(pixels, dtype=np.float32),
            header=header,
        ).writeto(temp_path, overwrite=True, output_verify="silentfix")
        os.replace(temp_path, resolved)
    finally:
        temp_path.unlink(missing_ok=True)
    return resolved


def _starun_roundtrip_shadow(
    image: np.ndarray,
    *,
    stretch_method: str,
    target_median: float,
    linked_stretch: bool,
    stat_bp_sigma: float,
    no_black_clip: bool,
):
    """Measure transform/inverse loss without changing inference decisions."""
    try:
        pixels = np.asarray(image, dtype=np.float32)
        if pixels.ndim == 2:
            rgb = np.repeat(pixels[..., None], 3, axis=2)
        elif pixels.ndim == 3 and pixels.shape[0] in (1, 3, 4) and pixels.shape[0] < min(pixels.shape[1:]):
            rgb = np.transpose(pixels[:3], (1, 2, 0))
            if rgb.shape[2] == 1:
                rgb = np.repeat(rgb, 3, axis=2)
        elif pixels.ndim == 3:
            rgb = pixels[:, :, :3]
            if rgb.shape[2] == 1:
                rgb = np.repeat(rgb, 3, axis=2)
        else:
            raise ValueError(f"unsupported image layout: {pixels.shape}")

        if stretch_method == "statistical":
            params = compute_stat_stretch_params(
                rgb,
                target_median=target_median,
                bp_sigma=stat_bp_sigma,
                no_black_clip=no_black_clip,
            )
            forward = apply_stat_stretch(rgb, params)
            restored = invert_stat_stretch(forward, params)
        elif stretch_method == "mtf":
            params = (
                _mtf_params_linked(rgb, targetbg=target_median)
                if linked_stretch
                else _mtf_params_unlinked(rgb, targetbg=target_median)
            )
            forward = apply_mtf_stretch(rgb, params)
            restored = apply_mtf_inverse(forward, params)
        elif stretch_method == "ihs":
            params = compute_ihs_per_channel_params(rgb, target=target_median)
            forward = apply_ihs_per_channel(rgb, params)
            restored = apply_ihs_per_channel_inverse(forward, params)
        else:
            forward = np.array(rgb, copy=True)
            restored = np.array(rgb, copy=True)

        error = np.abs(restored.astype(np.float32) - rgb.astype(np.float32))
        per_channel = []
        for channel in range(3):
            channel_error = error[..., channel]
            channel_forward = forward[..., channel]
            per_channel.append({
                "channel": channel,
                "mae": float(np.mean(channel_error)),
                "p99_abs": float(np.percentile(channel_error, 99.0)),
                "max_abs": float(np.max(channel_error)),
                "forward_low_clip_ratio": float(np.mean(channel_forward <= 0.0)),
                "forward_high_clip_ratio": float(np.mean(channel_forward >= 1.0)),
            })
        return {
            "status": "shadow",
            "mae": float(np.mean(error)),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "p99_abs": float(np.percentile(error, 99.0)),
            "max_abs": float(np.max(error)),
            "forward_low_clip_ratio": float(np.mean(forward <= 0.0)),
            "forward_high_clip_ratio": float(np.mean(forward >= 1.0)),
            "per_channel": per_channel,
        }
    except (TypeError, ValueError, FloatingPointError) as error:
        return {"status": "unavailable", "reason": str(error)}


def _write_file_mode_manifest(path: Path, payload: dict) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, resolved)
    finally:
        temp_path.unlink(missing_ok=True)


def _run_file_mode(args) -> int:
    """Run SyQon as a pure FITS file-to-file worker."""
    global ZENITH_MODEL_PATH

    if getattr(args, "axiom3", False):
        print(
            "Error: Starun file mode is Zenith-only; Axiom is not an approved asset",
            file=sys.stderr,
        )
        return 1

    input_path = Path(args.input_file).expanduser().resolve()
    starless_output = Path(args.starless_output).expanduser().resolve()
    starmask_output = (
        Path(args.starmask_output).expanduser().resolve()
        if args.starmask_output
        else None
    )
    if input_path == starless_output or (
        starmask_output is not None and input_path == starmask_output
    ):
        print("Error: SyQon file output must not overwrite the input FITS", file=sys.stderr)
        return 1

    runtime_engine_dir = Path.home() / ".siril" / "syqon_starless"
    try:
        engine_dir, model_source = resolve_zenith_model_dir(runtime_engine_dir)
        model_valid, model_message = validate_local_zenith_model(engine_dir)
    except (OSError, ValueError) as error:
        print(f"Error: Could not resolve local Zenith model: {error}", file=sys.stderr)
        return 1
    if not model_valid:
        print(f"Error: {model_message}", file=sys.stderr)
        return 1

    ZENITH_MODEL_PATH = engine_dir / "zenith.pt"
    selected_model = "zenith"
    print(f"Using local Zenith model from {model_source}: {ZENITH_MODEL_PATH}")
    print(model_message)

    try:
        (
            resolved_input,
            input_pixeldata,
            original_dtype,
            scale_factor,
            header_text,
        ) = _load_fits_file_input(input_path)
    except (OSError, ValueError, TypeError) as error:
        print(f"Error: Could not load input FITS: {error}", file=sys.stderr)
        return 1
    print(
        f"Loaded FITS file: {resolved_input} "
        f"{input_pixeldata.shape}, dtype: {original_dtype} -> float32"
    )
    roundtrip_shadow = _starun_roundtrip_shadow(
        input_pixeldata,
        stretch_method=args.stretch_method,
        target_median=args.target_median,
        linked_stretch=args.linked_stretch,
        stat_bp_sigma=args.stat_bp_sigma,
        no_black_clip=args.no_black_clip,
    )

    result_container = {"result": None, "error": None}

    def on_result_ready(result: tuple):
        result_container["result"] = result

    def on_error(error_message):
        result_container["error"] = str(error_message)

    engine = InferenceEngine(use_gpu=not args.no_gpu, model=selected_model)
    engine.process_async(
        image=input_pixeldata,
        tile_size=args.tile_size,
        overlap=args.overlap,
        generate_mask=True,
        mask_method=args.mask_method,
        use_amp=args.use_amp,
        stretch_method=args.stretch_method,
        target_median=args.target_median,
        linked_stretch=args.linked_stretch,
        stat_bp_sigma=args.stat_bp_sigma,
        no_black_clip=args.no_black_clip,
        callback=on_result_ready,
        error_callback=on_error,
    )
    print(f"Processing started with tile_size={args.tile_size}, overlap={args.overlap}...")
    import time
    while engine.is_processing():
        time.sleep(0.1)

    if result_container["error"]:
        print(
            f"Error: SyQon file inference failed: {result_container['error']}",
            file=sys.stderr,
        )
        return 1
    if result_container["result"] is None:
        print("Error: SyQon file inference produced no result", file=sys.stderr)
        return 1

    written = []
    try:
        starless, mask = result_container["result"]
        starless_restored = restore_image_dtype(
            starless,
            original_dtype,
            scale_factor,
        )
        mask_restored = (
            restore_image_dtype(mask, original_dtype, scale_factor)
            if mask is not None
            else None
        )
        if starmask_output is not None and mask_restored is None:
            raise ValueError("SyQon did not produce the requested starmask")

        written.append(
            _write_fits_file_output(
                starless_output,
                starless_restored,
                header_text,
                "starless",
            )
        )
        if starmask_output is not None and mask_restored is not None:
            written.append(
                _write_fits_file_output(
                    starmask_output,
                    mask_restored,
                    header_text,
                    "starmask",
                )
            )
    except (OSError, ValueError, TypeError) as error:
        for path in written:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        print(f"Error: SyQon file output failed: {error}", file=sys.stderr)
        return 1

    print(f"Saved starless: {starless_output}")
    if starmask_output is not None:
        print(f"Saved starmask: {starmask_output}")
    if args.manifest_output:
        _write_file_mode_manifest(
            Path(args.manifest_output),
            {
                "schema": "starun.syqon-worker.v1",
                "status": "accepted",
                "model": "zenith",
                "requested": {
                    "tile_size": int(args.tile_size),
                    "overlap": int(args.overlap),
                    "use_gpu": not bool(args.no_gpu),
                    "use_amp": bool(args.use_amp),
                    "stretch_method": str(args.stretch_method),
                    "target_median": float(args.target_median),
                    "linked_stretch": bool(args.linked_stretch),
                    "stat_bp_sigma": float(args.stat_bp_sigma),
                    "no_black_clip": bool(args.no_black_clip),
                    "mask_method": str(args.mask_method),
                },
                "actual": dict(STARUN_LAST_INFERENCE_DIAGNOSTICS),
                "shadow_metrics": {"transform_roundtrip": roundtrip_shadow},
            },
        )
    print("Processing complete!")
    return 0


# ============================================================================
# Main entry point
# ============================================================================
'''

FILE_MODE_ARGS_OLD = '''\
    parser.add_argument("--force_update_check", action="store_true",
        help="Force checking for updates to the Zenith model")
    args = parser.parse_args()

    # Connect to Siril
'''

FILE_MODE_ARGS_NEW = '''\
    parser.add_argument("--force_update_check", action="store_true",
        help="Force checking for updates to the Zenith model")
    parser.add_argument("--input-file", metavar="PATH",
        help="Read a FITS image directly instead of connecting to Siril")
    parser.add_argument("--starless-output", metavar="PATH",
        help="Write the starless FITS result directly to this path")
    parser.add_argument("--starmask-output", metavar="PATH",
        help="Write the starmask FITS result directly to this path")
    parser.add_argument("--manifest-output", metavar="PATH",
        help="Write the actual file-mode execution manifest to this path")
    parser.add_argument(
        "--stretch-method",
        choices=("statistical", "mtf", "ihs", "none"),
        default="statistical",
        help="Temporary inference stretch (default: statistical)",
    )
    parser.add_argument("--target-median", type=float, default=0.15,
        help="Temporary stretch target median (default: 0.15)")
    parser.add_argument("--stat-bp-sigma", type=float, default=5.0,
        help="Statistical stretch black-point sigma (default: 5.0)")
    parser.add_argument(
        "--mask-method",
        choices=("subtraction", "descreen"),
        default="subtraction",
        help="Star-mask decomposition method (default: subtraction)",
    )
    linked_group = parser.add_mutually_exclusive_group()
    linked_group.add_argument("--linked-stretch", action="store_true",
        dest="linked_stretch")
    linked_group.add_argument("--unlinked-stretch", action="store_false",
        dest="linked_stretch")
    black_group = parser.add_mutually_exclusive_group()
    black_group.add_argument("--no-black-clip", action="store_true",
        dest="no_black_clip")
    black_group.add_argument("--black-clip", action="store_false",
        dest="no_black_clip")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--use-amp", action="store_true", dest="use_amp")
    amp_group.add_argument("--no-amp", action="store_false", dest="use_amp")
    parser.set_defaults(linked_stretch=False, no_black_clip=False, use_amp=False)
    args = parser.parse_args()

    if not 0.01 <= args.target_median <= 0.50:
        parser.error("--target-median must be inside 0.01..0.50")
    if not 0.0 <= args.stat_bp_sigma <= 10.0:
        parser.error("--stat-bp-sigma must be inside 0..10")

    file_mode_values = (
        args.input_file,
        args.starless_output,
        args.starmask_output,
        args.manifest_output,
    )
    if any(file_mode_values):
        if not args.input_file or not args.starless_output:
            parser.error("--input-file and --starless-output must be used together")
        return _run_file_mode(args)

    # Connect to Siril
'''

CLI_SINGLE_ASYNC_OLD = '''\
            engine = InferenceEngine(use_gpu=with_gpu, model=selected_model)

            def on_result_ready(result: tuple):
                starless, mask = result
                starless_restored = restore_image_dtype(starless, original_dtype, scale_factor)
                mask_restored = restore_image_dtype(mask, original_dtype, scale_factor) if mask is not None else None
                try:
                    header_str = _make_safe_header(input_image.header, "starless")
                    filename   = siril.get_image_filename()
                    base       = os.path.splitext(os.path.basename(filename))[0]
                    base_starless = base if base.startswith("starless_") else "starless_" + base
                    new_filename  = os.path.join(cwd, base_starless)
                    siril.save_image_file(starless_restored, header_str, new_filename)
                    siril.undo_save_state("SyQon Starless - star removal")
                    try:
                        with siril.image_lock():
                            siril.set_image_pixeldata(starless_restored)
                            siril.set_image_metadata_from_header_string(header_str)
                        siril.set_image_filename(new_filename)
                    except Exception:
                        pass
                    if mask_restored is not None:
                        header_str = _make_safe_header(input_image.header, "starmask")
                        base_starmask = base if base.startswith("starmask_") else "starmask_" + base
                        new_filename  = os.path.join(cwd, base_starmask)
                        siril.save_image_file(mask_restored, header_str, new_filename)
                except Exception as e:
                    print(f"Could not send output to Siril: {e}")
                    sys.exit(1)

            def on_error(error_message):
                print(f"Error occurred: {error_message}", file=sys.stderr)
                sys.exit(1)

            engine.process_async(
                image=input_pixeldata,
                tile_size=args.tile_size,
                overlap=args.overlap,
                use_amp=False,
                callback=on_result_ready,
                error_callback=on_error,
            )
            print(f"Processing started with tile_size={args.tile_size}, overlap={args.overlap}...")
            import time
            while engine.is_processing():
                time.sleep(0.1)
            print("Processing complete!")
'''

CLI_SINGLE_ASYNC_NEW = '''\
            engine = InferenceEngine(use_gpu=with_gpu, model=selected_model)
            result_container = {"result": None, "error": None}

            def on_result_ready(result: tuple):
                result_container["result"] = result

            def on_error(error_message):
                result_container["error"] = str(error_message)

            engine.process_async(
                image=input_pixeldata,
                tile_size=args.tile_size,
                overlap=args.overlap,
                use_amp=False,
                callback=on_result_ready,
                error_callback=on_error,
            )
            print(f"Processing started with tile_size={args.tile_size}, overlap={args.overlap}...")
            import time
            while engine.is_processing():
                time.sleep(0.1)

            if result_container["error"]:
                print(
                    f"Error: SyQon inference failed: {result_container['error']}",
                    file=sys.stderr,
                )
                return 1
            if result_container["result"] is None:
                print("Error: SyQon inference produced no result", file=sys.stderr)
                return 1

            try:
                starless, mask = result_container["result"]
                starless_restored = restore_image_dtype(starless, original_dtype, scale_factor)
                mask_restored = restore_image_dtype(mask, original_dtype, scale_factor) if mask is not None else None
                header_str = _make_safe_header(input_image.header, "starless")
                filename   = siril.get_image_filename()
                base       = os.path.splitext(os.path.basename(filename))[0]
                base_starless = base if base.startswith("starless_") else "starless_" + base
                new_filename  = os.path.join(cwd, base_starless)
                siril.save_image_file(starless_restored, header_str, new_filename)
                siril.undo_save_state("SyQon Starless - star removal")
                try:
                    with siril.image_lock():
                        siril.set_image_pixeldata(starless_restored)
                        siril.set_image_metadata_from_header_string(header_str)
                    siril.set_image_filename(new_filename)
                except Exception:
                    pass
                if mask_restored is not None:
                    header_str = _make_safe_header(input_image.header, "starmask")
                    base_starmask = base if base.startswith("starmask_") else "starmask_" + base
                    new_filename  = os.path.join(cwd, base_starmask)
                    siril.save_image_file(mask_restored, header_str, new_filename)
            except Exception as e:
                print(f"Error: Could not send output to Siril: {e}", file=sys.stderr)
                return 1

            print("Processing complete!")
            return 0
'''

MAIN_EXIT_OLD = '''\
if __name__ == "__main__":
    main()
'''

MAIN_EXIT_NEW = '''\
if __name__ == "__main__":
    raise SystemExit(main())
'''

REPLACEMENTS = (
    (CONSTANTS_OLD, CONSTANTS_NEW, "constants"),
    (DOWNLOAD_START_OLD, DOWNLOAD_START_NEW, "download guard"),
    (UPDATE_CHECK_OLD, UPDATE_CHECK_NEW, "update guard"),
    (VERIFY_OLD, VERIFY_NEW, "local checksum validation"),
    (DOWNLOAD_BRANCH_OLD, DOWNLOAD_BRANCH_NEW, "download branch guard"),
    (MAIN_MODEL_OLD, MAIN_MODEL_NEW, "model directory selection"),
    (SAFE_HEADER_OLD, SAFE_HEADER_NEW, "canonical FITS header"),
    (RESTORE_DTYPE_OLD, RESTORE_DTYPE_NEW, "canonical float32 output"),
    (
        GUI_SEQUENCE_HEADER_OLD,
        GUI_SEQUENCE_HEADER_NEW,
        "GUI sequence canonical header",
    ),
    (CLI_SINGLE_HEADER_OLD, CLI_SINGLE_HEADER_NEW, "CLI single canonical header"),
    (
        CLI_SINGLE_MASK_HEADER_OLD,
        CLI_SINGLE_MASK_HEADER_NEW,
        "CLI single starmask canonical header",
    ),
    (
        CLI_SEQUENCE_HEADER_OLD,
        CLI_SEQUENCE_HEADER_NEW,
        "CLI sequence canonical header",
    ),
    (WEIGHT_CACHE_OLD, WEIGHT_CACHE_NEW, "single-axis tile blending weights"),
    (TILE_INFERENCE_OLD, TILE_INFERENCE_NEW, "Zenith safe tiling geometry"),
    (PROCESS_IMAGE_ARGS_OLD, PROCESS_IMAGE_ARGS_NEW, "process image parameters"),
    (STAT_STRETCH_CALL_OLD, STAT_STRETCH_CALL_NEW, "statistical stretch parameters"),
    (PROCESS_DIAGNOSTICS_OLD, PROCESS_DIAGNOSTICS_NEW, "inference diagnostics"),
    (PROCESS_ASYNC_ARGS_OLD, PROCESS_ASYNC_ARGS_NEW, "async parameters"),
    (PROCESS_ASYNC_FORWARD_OLD, PROCESS_ASYNC_FORWARD_NEW, "async forwarding"),
    (FILE_IO_HELPERS_OLD, FILE_IO_HELPERS_NEW, "file exchange helpers"),
    (FILE_MODE_ARGS_OLD, FILE_MODE_ARGS_NEW, "file mode arguments"),
    (CLI_SINGLE_ASYNC_OLD, CLI_SINGLE_ASYNC_NEW, "CLI error propagation"),
    (MAIN_EXIT_OLD, MAIN_EXIT_NEW, "main exit status propagation"),
)

REQUIRED_PATCH_TOKENS = (
    PATCH_SENTINEL,
    "resolve_zenith_model_dir",
    "validate_local_zenith_model",
    "syqon_network_downloads_allowed",
    "canonical Siril exchange domain",
    "SyQon output violates canonical 0..1 domain",
    'header_str = _make_safe_header(input_image.header, "starless")',
    'parser.add_argument("--input-file"',
    'parser.add_argument("--manifest-output"',
    "def _starun_pad_zenith_tensor",
    'return "solo"',
    "coverage_min",
    "stat_bp_sigma=stat_bp_sigma",
    "Starun file mode is Zenith-only",
    '"schema": "starun.syqon-worker.v1"',
    "def _run_file_mode(args) -> int:",
    "Error: SyQon file output failed",
    "raise SystemExit(main())",
)


def _resolve_target(target: Path) -> Path | None:
    if target.is_file():
        return target
    for relative in SYQON_RELATIVE_CANDIDATES:
        candidate = target / relative
        if candidate.is_file():
            return candidate
    return None


def apply_patch(target: Path) -> bool:
    script_path = _resolve_target(target)
    if script_path is None:
        return False
    text = script_path.read_text(encoding="utf-8")
    if all(token in text for token in REQUIRED_PATCH_TOKENS):
        return False
    if PATCH_SENTINEL not in text:
        upstream_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if upstream_sha256 != UPSTREAM_STARLESS_SHA256:
            raise RuntimeError(
                "refusing to patch an unpinned SyQon Starless.py: "
                f"commit={UPSTREAM_COMMIT}, expected={UPSTREAM_STARLESS_SHA256}, "
                f"actual={upstream_sha256}"
            )
    patched = text
    changed = False
    for old, new, label in REPLACEMENTS:
        if patched.count(new) == 1:
            continue
        count = patched.count(old)
        if count != 1:
            raise RuntimeError(
                f"cannot apply SyQon {label}: expected one anchor, found {count}"
            )
        patched = patched.replace(old, new, 1)
        changed = True

    if not all(token in patched for token in REQUIRED_PATCH_TOKENS):
        raise RuntimeError(f"incomplete SyQon offline patch: {script_path}")
    if not changed:
        return False

    compile(patched, str(script_path), "exec")
    script_path.write_text(patched, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    changed = apply_patch(args.target)
    print("SyQon offline model patch applied" if changed else "SyQon patch already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
