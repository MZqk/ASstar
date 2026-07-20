#!/usr/bin/env python3
"""Apply Seestar runtime fixes to a copied GraXpert-AI.py script.

The source files under resources/siril_plugins/vendor/siril-scripts are kept
as upstream files. This patcher is applied only to copied runtime/build trees.
"""

from __future__ import annotations

import argparse
from pathlib import Path


GRA_XPERT_RELATIVE_CANDIDATES = (
    Path("vendor/siril-scripts/processing/GraXpert-AI.py"),
    Path("vendor/siril-scripts/siril-scripts/processing/GraXpert-AI.py"),
    Path("processing/GraXpert-AI.py"),
    Path("GraXpert-AI.py"),
)


OLD_PRE_INFERENCE = """\
        print("Starting background extraction")

        # Handle different image formats
        was_mono = False
        if len(image.shape) == 2:
            # Handle grayscale image
            was_mono = True
            image = np.expand_dims(image, -1)
        # Convert to hwc format if needed:
        was_planar = False
        if image.shape[0] < 4 and len(image.shape) == 3 and image.shape[0] < image.shape[1] \\
                            and image.shape[0] < image.shape[2]:
            was_planar = True
            image = np.transpose(image, (1, 2, 0))

        # Store original shape for later reshaping
        original_shape = image.shape
        num_colors = image.shape[-1]
        if num_colors == 1:
            was_mono = True
        # Shrink and pad to avoid artifacts on borders
        padding = 8
        if progress_callback:
            progress_callback("Preparing image...", 0.05)
        # Resize to a standard size for the AI model
        imarray_shrink = cv2.resize(image, dsize=(256 - 2*padding, 256 - 2*padding),
                                    interpolation=cv2.INTER_LINEAR)

        if len(imarray_shrink.shape) == 2:
            imarray_shrink = np.expand_dims(imarray_shrink, -1)
        # Pad the image to avoid edge artifacts
        imarray_shrink = np.pad(imarray_shrink, ((padding, padding), (padding, padding), (0, 0)),
                            mode="edge")
        if progress_callback:
            progress_callback("Computing image statistics...", 0.1)

        # Calculate median and median absolute deviation for each channel
        median = []
        mad = []
        for c in range(num_colors):
            median.append(np.median(imarray_shrink[:, :, c]))
            mad.append(np.median(np.abs(imarray_shrink[:, :, c] - median[c])))
        if progress_callback:
            progress_callback("Normalizing image...", 0.15)

        # Normalize the image for the AI model
        imarray_shrink = (imarray_shrink - median) / mad * 0.04
        imarray_shrink = np.clip(imarray_shrink, -1.0, 1.0)
        # For grayscale, convert to RGB for the AI model
        if num_colors == 1:
            imarray_shrink = np.array([imarray_shrink[:, :, 0],
                                    imarray_shrink[:, :, 0],
                                    imarray_shrink[:, :, 0]])
            imarray_shrink = np.moveaxis(imarray_shrink, 0, -1)

        if progress_callback:
            progress_callback("Initializing ONNX runtime...", 0.25)

        # Initialize ONNX runtime session
        with s.SuppressedStderr():
            providers = onnx_helper.get_execution_providers_ordered(ai_gpu_acceleration)

            try:
                session = onnxruntime.InferenceSession(ai_path, providers=providers)
            except Exception as err:
                error_message = str(err)
                print("Warning: falling back to CPU.")
                if "cudaErrorNoKernelImageForDevice" in error_message \\
                    or "Error compiling model" in error_message:
                    print("ONNX cannot build an inferencing kernel for this GPU.")
                # Retry with CPU only
                providers = ['CPUExecutionProvider']
                try:
                    session = onnxruntime.InferenceSession(ai_path, providers=providers)
                except ONNXRuntimeError as err:
                    messagebox.showerror("Error", "Cannot build an inference model on this device")
                    return

        print(f"Using inference providers: {session.get_providers()}")

        if progress_callback:
            progress_callback("Running inference...", 0.4)

        # Run inference
        background, session = onnx_helper.run(session, ai_path, None, \\
                    {"gen_input_image": np.expand_dims(imarray_shrink, axis=0)})
        background = background[0][0]
"""


NEW_PRE_INFERENCE = """\
        print("Starting background extraction")

        # Handle different image formats
        was_mono = False
        if len(image.shape) == 2:
            # Handle grayscale image
            was_mono = True
            image = np.expand_dims(image, -1)
        # Convert to hwc format if needed:
        was_planar = False
        if image.shape[0] < 4 and len(image.shape) == 3 and image.shape[0] < image.shape[1] \\
                            and image.shape[0] < image.shape[2]:
            was_planar = True
            image = np.transpose(image, (1, 2, 0))

        # Store original shape for later reshaping
        original_shape = image.shape
        num_colors = image.shape[-1]
        if num_colors == 1:
            was_mono = True
        if progress_callback:
            progress_callback("Initializing ONNX runtime...", 0.1)

        # Initialize ONNX runtime early so the input size/order follows the model.
        with s.SuppressedStderr():
            providers = onnx_helper.get_execution_providers_ordered(ai_gpu_acceleration)
            try:
                session = onnxruntime.InferenceSession(ai_path, providers=providers)
            except Exception as err:
                error_message = str(err)
                print("Warning: falling back to CPU.")
                if "cudaErrorNoKernelImageForDevice" in error_message \\
                    or "Error compiling model" in error_message:
                    print("ONNX cannot build an inferencing kernel for this GPU.")
                providers = ['CPUExecutionProvider']
                try:
                    session = onnxruntime.InferenceSession(ai_path, providers=providers)
                except ONNXRuntimeError:
                    messagebox.showerror("Error", "Cannot build an inference model on this device")
                    return

        print(f"Using inference providers: {session.get_providers()}")

        input_shape = session.get_inputs()[0].shape
        is_nchw = True
        if len(input_shape) == 4 and input_shape[1] > 3:
            is_nchw = False
            target_h, target_w, target_c = input_shape[1], input_shape[2], input_shape[3]
        elif len(input_shape) == 4:
            target_h, target_w, target_c = input_shape[2], input_shape[3], input_shape[1]
        else:
            target_h, target_w, target_c = 256, 256, num_colors

        # Shrink and pad to avoid artifacts on borders
        padding = 8
        if progress_callback:
            progress_callback("Preparing image...", 0.15)
        imarray_shrink = cv2.resize(image, dsize=(target_w - 2*padding, target_h - 2*padding),
                                    interpolation=cv2.INTER_LINEAR)

        if len(imarray_shrink.shape) == 2:
            imarray_shrink = np.expand_dims(imarray_shrink, -1)
        if num_colors == 3 and target_c == 1:
            imarray_shrink = cv2.cvtColor(imarray_shrink, cv2.COLOR_RGB2GRAY)
            imarray_shrink = np.expand_dims(imarray_shrink, -1)
            num_colors = 1
        elif num_colors == 1 and target_c == 3:
            imarray_shrink = cv2.cvtColor(imarray_shrink, cv2.COLOR_GRAY2RGB)
            num_colors = 3

        # Pad the image to avoid edge artifacts
        imarray_shrink = np.pad(imarray_shrink, ((padding, padding), (padding, padding), (0, 0)),
                            mode="edge")
        if progress_callback:
            progress_callback("Computing image statistics...", 0.2)

        # Calculate median and median absolute deviation for each channel
        median = []
        mad = []
        for c in range(num_colors):
            median.append(np.median(imarray_shrink[:, :, c]))
            mad.append(np.median(np.abs(imarray_shrink[:, :, c] - median[c])))
        if progress_callback:
            progress_callback("Normalizing image...", 0.25)

        # Normalize the image for the AI model
        imarray_shrink = (imarray_shrink - median) / mad * 0.04
        imarray_shrink = np.clip(imarray_shrink, -1.0, 1.0)

        if progress_callback:
            progress_callback("Running inference...", 0.4)

        onnx_input = np.expand_dims(imarray_shrink, axis=0)
        if is_nchw:
            onnx_input = np.transpose(onnx_input, (0, 3, 1, 2))

        # Run inference
        background, session = onnx_helper.run(session, ai_path, None, \\
                    {"gen_input_image": onnx_input.astype(np.float32)},
                    return_first_output=True)
        # ONNXHelper normally returns a list of outputs. Request the first
        # output explicitly before removing the GraXpert batch dimension.
        background = np.asarray(background)
        if background.ndim == 4:
            if background.shape[0] != 1:
                raise RuntimeError(f"Unexpected GraXpert batch shape: {background.shape}")
            background = background[0]
        if is_nchw and background.ndim == 3:
            background = np.transpose(background, (1, 2, 0))
        elif background.ndim == 2:
            background = np.expand_dims(background, -1)
        if (background.ndim != 3 or background.size == 0
                or background.shape[0] <= 2 * padding
                or background.shape[1] <= 2 * padding
                or background.shape[2] not in (1, 3)):
            raise RuntimeError(f"Invalid GraXpert background shape: {background.shape}")
        if not np.all(np.isfinite(background)):
            raise RuntimeError("GraXpert background contains non-finite values")
"""


OLD_PATCHED_RUN_SUFFIX = """\
                    {"gen_input_image": onnx_input.astype(np.float32)})
"""


NEW_PATCHED_RUN_SUFFIX = """\
                    {"gen_input_image": onnx_input.astype(np.float32)},
                    return_first_output=True)
"""


OLD_PATCHED_OUTPUT_NORMALIZATION = """\
        # Remove only the batch dimension. GraXpert v2 BGE returns [1, 1, H, W];
        # removing both leading axes collapses it to 2D before HWC processing.
        background = background[0]
        if is_nchw and background.ndim == 3:
            background = np.transpose(background, (1, 2, 0))
        elif background.ndim == 2:
            background = np.expand_dims(background, -1)
"""


NEW_PATCHED_OUTPUT_NORMALIZATION = """\
        # ONNXHelper normally returns a list of outputs. Request the first
        # output explicitly before removing the GraXpert batch dimension.
        background = np.asarray(background)
        if background.ndim == 4:
            if background.shape[0] != 1:
                raise RuntimeError(f"Unexpected GraXpert batch shape: {background.shape}")
            background = background[0]
        if is_nchw and background.ndim == 3:
            background = np.transpose(background, (1, 2, 0))
        elif background.ndim == 2:
            background = np.expand_dims(background, -1)
        if (background.ndim != 3 or background.size == 0
                or background.shape[0] <= 2 * padding
                or background.shape[1] <= 2 * padding
                or background.shape[2] not in (1, 3)):
            raise RuntimeError(f"Invalid GraXpert background shape: {background.shape}")
        if not np.all(np.isfinite(background)):
            raise RuntimeError("GraXpert background contains non-finite values")
"""


OLD_PADDING = """\
        if padding != 0:
            background = background[padding:-padding, padding:-padding, :]
"""


NEW_PADDING = """\
        if padding != 0:
            if background.ndim == 3:
                background = background[padding:-padding, padding:-padding, :]
            else:
                background = background[padding:-padding, padding:-padding]
"""


def resolve_target(path: Path) -> Path:
    if path.is_file():
        return path
    for candidate in GRA_XPERT_RELATIVE_CANDIDATES:
        target = path / candidate
        if target.is_file():
            return target
    raise FileNotFoundError(f"GraXpert-AI.py not found under {path}")


def apply_patch(path: Path) -> bool:
    target = resolve_target(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    patched = text
    if OLD_PRE_INFERENCE in patched:
        patched = patched.replace(OLD_PRE_INFERENCE, NEW_PRE_INFERENCE, 1)
    else:
        # Upgrade runtime trees that already contain the first-generation
        # model-layout patch without recopying the upstream script.
        if OLD_PATCHED_RUN_SUFFIX in patched:
            patched = patched.replace(
                OLD_PATCHED_RUN_SUFFIX,
                NEW_PATCHED_RUN_SUFFIX,
                1,
            )
        if OLD_PATCHED_OUTPUT_NORMALIZATION in patched:
            patched = patched.replace(
                OLD_PATCHED_OUTPUT_NORMALIZATION,
                NEW_PATCHED_OUTPUT_NORMALIZATION,
                1,
            )
    if OLD_PADDING in patched:
        patched = patched.replace(OLD_PADDING, NEW_PADDING, 1)
    changed = patched != text
    if changed:
        target.write_text(patched, encoding="utf-8", newline="")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Plugin root or GraXpert-AI.py path")
    args = parser.parse_args()
    changed = apply_patch(args.path)
    print(f"GraXpert-AI runtime patch {'applied' if changed else 'already present'}: {resolve_target(args.path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
