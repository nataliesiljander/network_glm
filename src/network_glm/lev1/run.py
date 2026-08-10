#!/usr/bin/env python3
"""Level 1 GLM Analysis script using modular analysis package (CLI entry point).

This module is the CLI only; the work is split across sibling modules:
- :mod:`.spaces`  — analysis-space classification helpers
- :mod:`.prepare` — config/exclusions/dirs, file discovery, mask setup
- :mod:`.runner`  — per-run GLM processing + subject fixed effects
"""

import argparse
import logging
import sys
from pathlib import Path

from network_glm.lev1.prepare import (
    discover_and_validate_files,
    setup_analysis,
    setup_masks,
)
from network_glm.lev1.runner import (
    compute_fixed_effects_all,
    process_single_run,
)
from network_glm.task_config.loader import get_task_parameters
from network_glm import provenance

logger = logging.getLogger(__name__)

# fMRIPrep/BIDS file-type keys (in the discovered `files` dict) that represent
# ACTUAL study inputs consumed by the GLM, and should be hashed into the
# run-manifest. Masks are derived intermediates (re-created per run), so they
# are intentionally excluded.
_INPUT_FILE_KEYS = (
    "events",
    "confounds",
    "mni_data",
    "t1w_data",
    "left_surface",
    "right_surface",
    "cifti_bold",
)


def _positive_int(value: str) -> int:
    """Argparse type that accepts only integers >= 1."""
    iv = int(value)
    if iv < 1:
        raise argparse.ArgumentTypeError("--min-runs must be >= 1")
    return iv


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger for the analysis pipeline.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def get_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description="Level 1 GLM Analysis for Network R01 dataset")
    parser.add_argument("--subj-id", type=str, required=True, help="Subject ID")
    parser.add_argument("--task-name", type=str, required=True, help="Task name")
    parser.add_argument("--bids-dir", type=str, required=True, help="BIDS directory path")
    parser.add_argument("--fmriprep-dir", type=str, required=True, help="fMRIPrep directory path")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=False,
        default="./results/",
        help="GLM results directory",
    )
    parser.add_argument(
        "--space",
        choices=["T1w", "MNI", "surface", "fsaverage6", "fsLR"],
        default="MNI",
        help="Analysis space. T1w/MNI for volumetric; surface for fsnative; "
        "fsaverage6 for fsaverage6 GIFTI; fsLR for fsLR den-91k CIFTI",
    )
    parser.add_argument(
        "--within-subject-threshold",
        type=float,
        default=1.0,
        help="Threshold for mask intersection (0.0-1.0)",
    )
    parser.add_argument(
        "--exclusions-file",
        type=str,
        required=True,
        help="Path to exclusions JSON file",
    )
    parser.add_argument(
        "--residuals",
        action="store_true",
        default=False,
        help="Compute residuals (default: false)",
    )
    # TODO: Consider removing smoothing if downstream analyses do not
    # require it (added per Du et al. 2025, Neuron).  For surface space
    # this calls FreeSurfer mri_surf2surf (module load biology
    # freesurfer/8.1.0).
    parser.add_argument(
        "--smoothing-fwhm",
        type=float,
        default=None,
        help="Spatial smoothing FWHM in mm applied to BOLD before GLM "
        "(affects all outputs). None means no smoothing.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip runs where residual files already exist (useful for resuming)",
    )
    parser.add_argument(
        "--fc-confounds",
        action="store_true",
        default=False,
        help="Regress tissue confounds (global signal, WM, CSF) from residuals "
        "for FC analysis. Requires --residuals. Follows Du et al. 2025.",
    )
    parser.add_argument(
        "--no-residual-filter",
        action="store_true",
        help="Emit residuals WITHOUT the 0.01-0.1 Hz band-pass (fsLR/CIFTI path); "
        "defer temporal filtering to downstream (e.g. XCP-D).",
    )
    parser.add_argument(
        "--confounds-mode",
        choices=["full", "no-motion", "task-only"],
        default="full",
        help="Nuisance regressors in the lev1 design: full (cosine+24p motion+spikes), "
        "no-motion (cosine only), task-only (none). NSI-experiment arms.",
    )
    parser.add_argument(
        "--mni-template",
        default="MNI152NLin6Asym",
        help="fMRIPrep MNI template name for --space MNI " "(default: MNI152NLin6Asym)",
    )
    parser.add_argument(
        "--mni-res",
        default="2",
        help="Resolution suffix for --space MNI (default: 2)",
    )
    parser.add_argument(
        "--min-runs",
        type=_positive_int,
        default=2,
        help="Minimum runs required to compute a non-tagged fixed-effects map. "
        "Below this threshold, the saved map is tagged _desc-belowMinRuns "
        "and lev2 will filter it out (default: 2).",
    )
    parser.add_argument(
        "--skip-qc-plots",
        action="store_true",
        default=False,
        help="Skip per-contrast surface QC plots (matplotlib renders ~10 plots "
        "per hemisphere per run; for a 46-subject cohort this adds many hours "
        "of wall time with no impact on the science). The contrast .func.gii "
        "files are still saved and can be re-plotted offline.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        default=False,
        help="Permit recording provenance against an uncommitted (dirty) git "
        "working tree without warning. Without this flag a dirty tree warns "
        "loudly to stderr but the run still proceeds; the manifest records "
        "code_dirty truthfully either way.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging",
    )
    parser.add_argument(
        "--add-ssd-regressor",
        action="store_true",
        default=False,
        help="Add demeaned stop-signal-delay regressors (output_all and output_sf)"
        "to stopSignal task events. No effect on other tasks",
    )
    return parser


def _collect_run_inputs(files):
    """Flatten the discovered ``files`` dict into the list of ACTUAL inputs.

    ``files`` is ``{session: {run: {file_type: Path}}}`` (see
    :class:`~network_glm.io.file_discovery.FileFinder`). We hash the
    study inputs the GLM actually consumes — events, confounds, and the BOLD
    timeseries for whichever space ran (``mni_data`` / ``t1w_data`` /
    ``left_surface`` + ``right_surface`` / ``cifti_bold``). Brain masks are
    derived intermediates (re-created per run) and are intentionally omitted.

    Returns a de-duplicated, deterministically sorted list of Paths.
    """
    collected: set = set()
    for runs in files.values():
        for run_files in runs.values():
            for key in _INPUT_FILE_KEYS:
                path = run_files.get(key)
                if path is not None:
                    collected.add(Path(path))
    return sorted(collected, key=str)


def _write_lev1_provenance(results_dir, args, dirs, input_files):
    """Write additive provenance for one subject×task lev1 invocation.

    - ``dataset_description.json`` at the shared ``results_dir`` (idempotent:
      many subject×task invocations share one results_dir; rewriting an
      identical file is fine and BIDS-valid).
    - ``run-manifest.json`` at the per-subject×task output subdir
      (``dirs['base']``), recording stage='lev1', the actual inputs consumed,
      and the compiled-exclusions source.

    Called AFTER scientific outputs are written so a manifest error never costs
    science; the error is allowed to surface (fail loud) rather than be
    swallowed. ``allow_dirty`` is threaded from the CLI flag.
    """
    allow_dirty = getattr(args, "allow_dirty", False)
    provenance.write_dataset_description(
        results_dir,
        name="lev1",
        source_datasets=[
            {"URL": str(args.bids_dir)},
            {"URL": str(args.fmriprep_dir)},
        ],
    )
    provenance.write_run_manifest(
        dirs["base"],
        stage="lev1",
        args=args,
        inputs=input_files,
        exclusions_source=args.exclusions_file,
        allow_dirty=allow_dirty,
    )


def main(argv=None):
    """Run level 1 analysis with command line arguments."""
    parser = get_parser()
    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose)

    # Provenance is ADDITIVE: warn loudly (but do not fail) when stamping a
    # dirty tree, unless --allow-dirty. The manifest records code_dirty truly.
    if provenance.git_is_dirty() and not args.allow_dirty:
        print(
            "WARNING: git working tree is dirty; lev1 provenance will record "
            "code_dirty=true. Commit/stash for a reproducible stamp, or pass "
            "--allow-dirty to silence this warning.",
            file=sys.stderr,
        )

    # Setup
    # setup_analysis still returns expected_sessions/exclusions_by_type; they are
    # not consumed downstream (the active exclusion set is `exclusions`), so bind
    # them to throwaways here rather than thread dead args onward.
    config, sample_type, _expected_sessions, exclusions, _exclusions_by_type, dirs = setup_analysis(
        args
    )

    # File discovery
    files = discover_and_validate_files(config, args)

    # Masks
    combined_mask_path = setup_masks(files, args, dirs)

    # Get task parameters
    task_params = get_task_parameters(args.task_name)

    # Process each run
    run_count = 0
    failed_runs = []

    for session in sorted(files.keys()):
        for run in sorted(files[session].keys()):
            run_count += 1
            try:
                success = process_single_run(
                    session,
                    run,
                    files[session][run],
                    args,
                    sample_type,
                    dirs,
                    task_params,
                    exclusions,
                )
                if not success:
                    failed_runs.append(f"{session}/{run}")
            except Exception as e:
                logger.error("Failed to process %s/%s: %s", session, run, e)
                failed_runs.append(f"{session}/{run}")

    # Fixed effects (compute even with partial failures)
    compute_fixed_effects_all(
        args,
        dirs,
        exclusions,
        combined_mask_path,
        failed_runs,
        run_count,
    )

    # Provenance (ADDITIVE) — written AFTER all scientific outputs so a manifest
    # error never loses science. Errors are allowed to surface (fail loud).
    _write_lev1_provenance(
        Path(args.results_dir),
        args,
        dirs,
        _collect_run_inputs(files),
    )

    # Summary
    successful_runs = run_count - len(failed_runs)
    logger.info("Analysis complete: %d/%d runs successful", successful_runs, run_count)
    if failed_runs:
        logger.warning("Failed runs: %s", ", ".join(failed_runs))

    return 1 if len(failed_runs) > 0 else 0


if __name__ == "__main__":
    exit(main())
