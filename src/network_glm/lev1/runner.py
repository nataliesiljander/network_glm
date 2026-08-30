"""Per-run processing + fixed-effects orchestration for the lev1 GLM pipeline.

Consumes the context built by :mod:`.prepare` and drives one run at a time
through design-matrix construction, QC, GLM fit, contrast/residual saving
(volumetric via ``process_volumetric_run``, surface via
``process_surface_run``), then aggregates runs into subject fixed effects
(``compute_fixed_effects_all``). The CLI entry point lives in :mod:`.run`.
"""

import logging
import tempfile
from pathlib import Path

import pandas as pd

from network_glm.exclusions import load_contrast_exclusions
from network_glm.lev1.processing.cifti_io import load_dtseries
from network_glm.lev1.processing.confounds import (
    get_fc_confounds,
    load_and_process_confounds,
)
from network_glm.lev1.processing.contrasts import (
    compute_run_contrasts,
    filter_contrasts_for_dropped_columns,
)
from network_glm.lev1.processing.design import create_design_matrix
from network_glm.lev1.processing.events import (
    add_junk_trials,
    load_bold_data_with_dummy_removal,
    preprocess_events,
    save_simplified_events,
    stop_fail_violation,
)
from network_glm.lev1.processing.fixed_effects import compute_subject_fixed_effects
from network_glm.lev1.processing.glm import (
    fit_run_glm,
    handle_zero_variance_columns,
    validate_design_matrix,
    validate_glm_inputs,
)
from network_glm.lev1.processing.quality_control import run_quality_control
from network_glm.lev1.processing.residuals import (
    cifti_residual_filename,
    process_cifti_residuals,
    process_run_residuals,
    process_surface_residuals,
    surface_residual_filename,
)
from network_glm.lev1.processing.surface_data import (
    SurfaceGLM,
    find_freesurfer_subjects_dir,
    get_surface_scan_info,
    load_surface_data,
    plot_surface_stat_map,
    resolve_freesurfer_subject,
    smooth_surface_gifti,
)
from network_glm.lev1.spaces import is_cifti_space, is_surface_space, resolve_surface_space
from network_glm.task_config.loader import get_task_contrasts

logger = logging.getLogger(__name__)


def process_volumetric_run(
    bold_data,
    design_matrix,
    contrasts,
    run_files,
    args,
    dirs,
    base_filename,
    tr,
    mask_key,
    compute_residuals,
    fc_confounds=None,
):
    """Fit volumetric GLM and compute contrasts (and optional residuals).

    ``fc_confounds``, when provided, is regressed from the post-GLM residuals
    via nilearn.signal.clean — matching the surface path so that
    ``--fc-confounds`` produces FC-quality residuals in either space.

    Returns:
        Dict of contrast results.
    """
    validation = validate_glm_inputs(bold_data, design_matrix, run_files[mask_key])
    if not validation["is_valid"]:
        raise ValueError(f'GLM validation failed: {validation["errors"]}')

    run_mask = run_files[mask_key]

    # When residuals are requested, fit once with minimize_memory=False
    # so the same model can be used for both contrasts and residuals.
    analysis_type = "residual" if compute_residuals else "task"
    fitted_glm = fit_run_glm(
        bold_data,
        design_matrix,
        analysis_type,
        args.subj_id,
        tr,
        smoothing_fwhm=args.smoothing_fwhm,
        mask_img=run_mask,
    )

    contrast_results = compute_run_contrasts(
        fitted_glm,
        args.task_name,
        dirs["indiv_contrasts"],
        base_filename,
        contrasts=contrasts,
    )
    logger.info("Saved %d contrasts", len(contrast_results))

    # Process residuals if requested
    if compute_residuals:
        residuals_result = process_run_residuals(
            fitted_glm,
            dirs["task_residuals"],
            base_filename,
            tr,
            mask_img=run_mask,
            fc_confounds=fc_confounds,
        )
        if not residuals_result["success"]:
            logger.warning("Residuals processing had issues: %s", residuals_result["errors"])

    return contrast_results


def process_surface_run(
    run_files,
    design_matrix,
    contrasts,
    args,
    dirs,
    base_filename,
    tr,
    dummy_scans,
    compute_residuals=False,
    surface_space="fsnative",
    fc_confounds=None,
):
    """Fit surface GLM per hemisphere and compute contrasts (and optional residuals).

    Returns:
        Dict mapping hemispheres to contrast results.
    """
    all_hemisphere_results = {}

    # Find FreeSurfer subjects dir + resolve the FS-subject name used by
    # mri_surf2surf. The choice depends on the BOLD's surface space:
    #
    #   fsaverage / fsaverage6  -> use 'fsaverage6' (40962 v/hemi). The BOLD
    #                              has been resampled to the group template;
    #                              smoothing must operate on that mesh.
    #   fsnative                -> use the per-subject FS recon (resolved via
    #                              `resolve_freesurfer_subject` because
    #                              fmriprep's longitudinal anat workflow
    #                              names recons `sub-X_ses-Y`).
    #
    # Passing the per-subject recon while smoothing fsaverage6 BOLD produces
    # the dimension-mismatch error (e.g. 131403 vs 40962 vertices).
    subjects_dir = None
    fs_subject = args.subj_id
    if args.smoothing_fwhm is not None:
        subjects_dir = find_freesurfer_subjects_dir(Path(args.fmriprep_dir))
        if subjects_dir is None:
            raise FileNotFoundError("Cannot find FreeSurfer subjects dir for surface smoothing")
        if surface_space in ("fsaverage", "fsaverage6"):
            fs_subject = "fsaverage6"
        else:
            fs_subject = resolve_freesurfer_subject(args.subj_id, subjects_dir)

    for hemisphere, bold_key in [("L", "left_surface"), ("R", "right_surface")]:
        bold_path = run_files[bold_key]
        logger.info("Processing hemisphere %s...", hemisphere)

        # Apply spatial smoothing to BOLD if requested
        if args.smoothing_fwhm is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                smoothed_path = Path(tmp_dir) / f"smoothed_hemi-{hemisphere}.func.gii"
                smooth_surface_gifti(
                    bold_path,
                    smoothed_path,
                    fs_subject,
                    hemisphere,
                    args.smoothing_fwhm,
                    subjects_dir,
                )
                surface_data = load_surface_data(smoothed_path, dummy_scans=dummy_scans)
        else:
            surface_data = load_surface_data(bold_path, dummy_scans=dummy_scans)
        logger.debug("Surface data shape: %s", surface_data.shape)

        # Validate inputs before fitting. The volumetric branch (process_volumetric_run)
        # has called validate_glm_inputs for a long time; the surface branch
        # historically had no equivalent and would have silently propagated NaN /
        # mis-shaped designs through nilearn run_glm into garbage contrast maps.
        # We validate once per hemisphere because each hemisphere produces its own
        # surface_data and the row-count check needs that array's first dim.
        validation = validate_design_matrix(design_matrix, n_scans=surface_data.shape[0])
        if not validation["is_valid"]:
            raise ValueError(
                f'Surface GLM validation failed (hemi-{hemisphere}): ' f'{validation["errors"]}')

        surface_glm = SurfaceGLM(t_r=tr)
        surface_glm.fit(surface_data, design_matrix)

        # Compute + save contrasts via the shared saver (RF-6). compute_run_contrasts
        # already supports surface output through its ``hemisphere`` arg — byte-identical
        # naming (_hemi-H_contrast-..._rtmodel-RTDur_stat-...func.gii), the same three
        # stat maps, and the same return structure as the volumetric path. The float32
        # recast is a no-op for GIFTI (cast_nifti_to_float32(is_surface=True)).
        contrast_results = compute_run_contrasts(
            surface_glm,
            args.task_name,
            dirs["indiv_contrasts"],
            base_filename,
            contrasts=contrasts,
            hemisphere=hemisphere,
        )
        logger.info("Saved %d contrasts for hemisphere %s", len(contrast_results), hemisphere)

        # Generate QC plots (skipped under --skip-qc-plots; matplotlib renders
        # are slow at cohort scale — ~10 plots × 2 hemis × N runs adds many
        # hours of wall time per subject. The .func.gii files are persisted
        # above and can be re-plotted offline if review is needed.)
        if getattr(args, "skip_qc_plots", False):
            logger.debug("Skipping QC plots for hemisphere %s (--skip-qc-plots)", hemisphere)
            continue
        qc_count = 0
        for contrast_name, paths in contrast_results.items():
            try:
                qc_filename = (
                    f"{base_filename}_hemi-{hemisphere}" f"_contrast-{contrast_name}_qc.png"
                )
                qc_path = dirs["quality_control"] / qc_filename
                title = f"{args.subj_id} - {contrast_name} (hemi-{hemisphere})"
                plot_surface_stat_map(
                    paths["effect_size"],
                    qc_path,
                    hemisphere,
                    title=title,
                    fmriprep_dir=Path(args.fmriprep_dir),
                    subject_id=args.subj_id,
                )
                qc_count += 1
            except Exception as e:
                logger.debug("Failed to plot %s: %s", contrast_name, e)
        logger.debug("Saved %d QC plots for hemisphere %s", qc_count, hemisphere)

        # Process surface residuals if requested
        if compute_residuals:
            process_surface_residuals(
                surface_glm,
                dirs["task_residuals"],
                base_filename,
                hemisphere,
                tr,
                fc_confounds=fc_confounds,
                surface_space=surface_space,
            )

        all_hemisphere_results[hemisphere] = contrast_results

    return all_hemisphere_results


def process_cifti_run(run_files, design_matrix, args, dirs, base_filename, tr, fc_confounds=None):
    """Fit a GLM over fsLR den-91k grayordinates and write residuals as a dtseries."""
    if not getattr(args, "residuals", False):
        raise ValueError("--space fsLR is residuals-only; pass --residuals.")
    data, template = load_dtseries(run_files["cifti_bold"])  # (T, 91282)
    validation = validate_design_matrix(design_matrix, n_scans=data.shape[0])
    if not validation["is_valid"]:
        raise ValueError(f"CIFTI GLM design validation failed: {validation['errors']}")
    glm = SurfaceGLM(t_r=tr).fit(data, design_matrix)
    no_filter = getattr(args, "no_residual_filter", False)
    lp = None if no_filter else 0.1
    hp = None if no_filter else 0.01
    return process_cifti_residuals(
        glm, template, dirs["task_residuals"], base_filename, tr,
        low_pass=lp, high_pass=hp, fc_confounds=fc_confounds,
    )


def _run_base_filename(subj_id, session, task_name, run):
    """BIDS-style per-run base filename shared by run keys + output filenames."""
    return f"{subj_id}_{session}_task-{task_name}_{run}"


def process_single_run(session, run, run_files, args, sample_type, dirs, task_params, exclusions):
    """Process a single run (volumetric or surface).

    Returns:
        True if successful, False if failed.
    """
    tr = task_params["tr"]
    run_key = _run_base_filename(args.subj_id, session, args.task_name, run)

    if run_key in exclusions:
        logger.info("Skipping excluded run: %s/%s", session, run)
        return True  # Not a failure, just skipped

    # Skip if all output files already exist
    if args.skip_existing:
        base_filename = _run_base_filename(args.subj_id, session, args.task_name, run)
        if is_surface_space(args.space) and args.residuals:
            surface_space = resolve_surface_space(args.space)
            lh_res = dirs["task_residuals"] / surface_residual_filename(
                base_filename, "L", surface_space
            )
            rh_res = dirs["task_residuals"] / surface_residual_filename(
                base_filename, "R", surface_space
            )
            if lh_res.exists() and rh_res.exists():
                logger.info("Skipping %s (outputs already exist)", run_key)
                return True
        elif is_cifti_space(args.space) and args.residuals:
            cifti_res = dirs["task_residuals"] / cifti_residual_filename(base_filename)
            if cifti_res.exists():
                logger.info("Skipping %s (outputs already exist)", run_key)
                return True
        elif not is_surface_space(args.space) and args.residuals:
            vol_res = dirs["task_residuals"] / f"{base_filename}_task-regressed-residuals.nii.gz"
            if vol_res.exists():
                logger.info("Skipping %s (outputs already exist)", run_key)
                return True

    logger.info("Processing %s/%s...", session, run)

    # Load BOLD data or get scan count
    if is_cifti_space(args.space):
        if "cifti_bold" not in run_files:
            raise ValueError(f"Missing cifti_bold for {session}/{run}")
        import nibabel as nib

        n_scans = nib.load(str(run_files["cifti_bold"])).shape[0]
    elif is_surface_space(args.space):
        if "left_surface" not in run_files or "right_surface" not in run_files:
            raise ValueError(f"Missing surface files for {session}/{run}")
        n_scans_total, _ = get_surface_scan_info(run_files["left_surface"])
        # BOLD is already trimmed by trim_bold.py; do not remove dummy scans again
        n_scans = n_scans_total
    else:
        mask_key = f"{args.space.lower()}_brain_mask"
        data_key = f"{args.space.lower()}_data"
        if mask_key not in run_files or data_key not in run_files:
            raise ValueError(f"Missing required files for {session}/{run}")
        # BOLD is already trimmed by trim_bold.py; load without further removal
        bold_data = load_bold_data_with_dummy_removal(run_files[data_key], dummy_scans=0)
        n_scans = bold_data.shape[3]

    # Load and preprocess events
    # Onsets are already adjusted for dummy scans during event file creation
    # (shifted by -7*1.49s = -10.43s); do not adjust again
    events_df = pd.read_csv(run_files["events"], sep="\t")
    processed_events = preprocess_events(events_df, args.task_name, n_scans=n_scans, tr=tr)
    processed_events_with_junk, percent_junk = add_junk_trials(processed_events, args.task_name)

    task_name_norm = (args.task_name or "").strip().lower()
    #Scope note: task-name gating for the stop_fail_violation regressor 
    #does not extend to the dual tasks
    if task_name_norm in ("stopsignal", "stop_signal"):
        processed_events_with_junk = stop_fail_violation(processed_events_with_junk)
        processed_events_with_junk = add_ssd_regressor(processed_events_with_junk)

    # Load confounds. BOLD is pre-trimmed by scripts/trim_bold.py and fMRIPrep
    # is run with --dummy-scans 0, so the confounds TSV already matches the
    # trimmed BOLD length. Do not trim confounds further.
    selected_confounds = load_and_process_confounds(
        run_files["confounds"], args.task_name, sample_type, dummy_scans=0,
        confounds_mode=getattr(args, "confounds_mode", "full"),
    )
    if len(selected_confounds) != n_scans:
        raise ValueError(f"Confounds length mismatch: {len(selected_confounds)} != {n_scans}")

    # Create design matrix
    design_matrix, regressor_3cols = create_design_matrix(
        processed_events_with_junk,
        selected_confounds,
        args.task_name,
        n_scans,
        tr,
    )
    logger.debug("Design matrix shape: %s", design_matrix.shape)

    # Handle zero-variance columns
    design_matrix, dropped_columns = handle_zero_variance_columns(design_matrix)

    # Get and filter contrasts
    all_contrasts = get_task_contrasts(args.task_name)
    contrasts, skipped_contrasts = filter_contrasts_for_dropped_columns(
        all_contrasts, dropped_columns
    )

    # Quality control
    vifs, qa_failed = run_quality_control(
        design_matrix,
        contrasts,
        percent_junk,
        dirs["quality_control"],
        subject_id=args.subj_id,
        session=session,
        run=run,
        task_name=args.task_name,
    )

    # Save simplified events
    if regressor_3cols:
        simplified_events_file = (
            dirs["simplified_events"]
            / f"{_run_base_filename(args.subj_id, session, args.task_name, run)}_desc-simplifiedEvents.csv"
        )
        save_simplified_events(regressor_3cols, simplified_events_file)

    if qa_failed:
        logger.error("Skipping GLM fitting due to QA failure")
        return False

    base_filename = _run_base_filename(args.subj_id, session, args.task_name, run)

    # Load FC confounds once if requested — used by both surface and
    # volumetric residual paths so `--fc-confounds` has identical semantics
    # in either space (previously volumetric silently ignored the flag).
    fc_confounds = None
    if args.residuals and args.fc_confounds:
        confounds_df = pd.read_csv(run_files["confounds"], sep="\t", na_values=["n/a"]).fillna(0)
        fc_confounds_df = get_fc_confounds(confounds_df)
        if not fc_confounds_df.empty:
            # BOLD is pre-trimmed and fMRIPrep runs with --dummy-scans 0,
            # so confounds TSV already matches trimmed BOLD length.
            fc_confounds = fc_confounds_df.values
            logger.info("FC confounds: %d columns", fc_confounds.shape[1])

    if is_cifti_space(args.space):
        process_cifti_run(
            run_files,
            design_matrix,
            args,
            dirs,
            base_filename,
            tr,
            fc_confounds=fc_confounds,
        )
    elif is_surface_space(args.space):
        surface_space = resolve_surface_space(args.space)

        process_surface_run(
            run_files,
            design_matrix,
            contrasts,
            args,
            dirs,
            base_filename,
            tr,
            0,  # BOLD already trimmed
            compute_residuals=args.residuals,
            surface_space=surface_space,
            fc_confounds=fc_confounds,
        )
    else:
        compute_residuals = args.residuals
        process_volumetric_run(
            bold_data,
            design_matrix,
            contrasts,
            run_files,
            args,
            dirs,
            base_filename,
            tr,
            mask_key,
            compute_residuals,
            fc_confounds=fc_confounds,
        )

    return True


def compute_fixed_effects_all(
    args,
    dirs,
    exclusions,
    combined_mask_path,
    failed_runs,
    run_count,
):
    """Compute fixed effects across runs, supporting partial-run analysis.

    Tags output with desc-partialRuns if any runs failed.
    """
    if is_cifti_space(args.space):
        logger.info("Skipping fixed-effects for CIFTI/fsLR (residuals-only path)")
        return

    # Compute fixed effects on available successful runs (partial run support)
    successful_runs = run_count - len(failed_runs)
    if successful_runs == 0:
        logger.error("No successful runs - skipping fixed effects")
        return

    if failed_runs:
        logger.warning(
            "Computing fixed effects on %d/%d successful runs (partial)",
            successful_runs,
            run_count,
        )

    # Per-contrast VIF exclusions (lev1_outlier 'exclude-contrast'): drop a single
    # run's contribution to one contrast's fixed-effects (not the whole run).
    contrast_exclusions = load_contrast_exclusions(args.exclusions_file)
    if contrast_exclusions:
        logger.info("Loaded %d per-contrast exclusions", len(contrast_exclusions))

    logger.info("Computing fixed effects...")
    try:
        if is_surface_space(args.space):
            surface_space = resolve_surface_space(args.space)
            for hemisphere in ["L", "R"]:
                logger.info("Fixed effects for hemisphere %s...", hemisphere)
                results = compute_subject_fixed_effects(
                    args.subj_id,
                    args.task_name,
                    dirs["indiv_contrasts"],
                    dirs["fixed_effects"],
                    mask_img=None,
                    exclusions=exclusions,
                    min_runs=args.min_runs,
                    hemisphere=hemisphere,
                    surface_space=surface_space,
                    contrast_exclusions=contrast_exclusions,
                )
                logger.info("Fixed effects: %d contrasts (hemi-%s)", len(results), hemisphere)
        else:
            results = compute_subject_fixed_effects(
                args.subj_id,
                args.task_name,
                dirs["indiv_contrasts"],
                dirs["fixed_effects"],
                combined_mask_path,
                exclusions,
                min_runs=args.min_runs,
                contrast_exclusions=contrast_exclusions,
            )
            logger.info("Fixed effects: %d contrasts", len(results))
    except Exception as e:
        logger.error("Fixed effects computation failed: %s", e)
