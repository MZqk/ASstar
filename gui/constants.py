"""Shared constants for the Seestar GUI."""

FITS_SUFFIXES = (".fit", ".fits")
LIGHT_FRAME_EXPANSION_FACTOR = 3.0
LIGHT_PREPROCESS_SEQUENCE_COPIES = 2.0
# Peak FITS/output copies, calibrated against the current stage artifact contract.
# A Stage 2 resume currently produces 37 process FITS plus final export headroom;
# full stacked mode adds the Stage 1/2 working copies, while linear resume starts
# at star separation and therefore needs a smaller budget.
STACKED_STAGE_ARTIFACT_COPIES = 44.0
STAGE2_RESUME_STAGE_ARTIFACT_COPIES = 40.0
LINEAR_RESUME_STAGE_ARTIFACT_COPIES = 28.0
DISK_SPACE_HEADROOM_RATIO = 0.15
DISK_SPACE_MIN_HEADROOM_BYTES = 1 * 1024 * 1024 * 1024
RUNTIME_DEPENDENCY_EXPANSION_FACTOR = 3.25
RUNTIME_DISK_MIN_HEADROOM_BYTES = 512 * 1024 * 1024
