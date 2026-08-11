"""Event processing pipeline for first-level GLM analysis.

This module handles preprocessing of BIDS events.tsv files before they are
used to construct GLM design matrices. The pipeline consists of:

1. Raw events.tsv loaded from BIDS directory
2. String "n/a" values converted to numeric NaN
3. Negative response times marked as junk and set to NaN
4. Nuisance trial columns (omission, commission, rt_fast) computed

Note: Onset adjustment for dummy scans and break_with_performance_feedback
labeling are handled upstream during event file creation (events/create.py).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from network_glm.task_config.loader import TR

logger = logging.getLogger(__name__)

# Constants
MIN_RT = 0.2  # Minimum valid response time in seconds


def preprocess_events(
    events_df: pd.DataFrame,
    task_name: str,
    adjust_for_dummy_scans: bool = False,
    dummy_scans: int = 0,
    tr: float = TR,
    n_scans: int | None = None,
) -> pd.DataFrame:
    """Preprocess events dataframe for GLM modeling.

    Converts "n/a" strings to NaN, marks negative response times as junk,
    and adds constant/junk columns needed by the design matrix.

    Onset adjustment and break_with_performance_feedback labeling are
    handled upstream during event file creation (events/create.py).
    The adjust_for_dummy_scans parameter defaults to False since onsets
    are already adjusted.

    Args:
        events_df: Events dataframe from BIDS events.tsv
        task_name: Name of the task
        adjust_for_dummy_scans: Whether to adjust onsets (default False — already done)
        dummy_scans: Number of dummy scans (default 0 — already trimmed)
        tr: Repetition time in seconds
        n_scans: Total BOLD timepoints. When set, drop rows whose
            ``onset >= n_scans * tr`` (handles salvaged scans where the BOLD
            is shorter than the original behavioral session).

    Returns:
        Preprocessed events dataframe with additional columns.
    """
    events_df = events_df.copy()

    # Convert columns that may contain "n/a" strings to numeric
    for col in ["onset", "duration", "response_time", "key_press", "correct_response"]:
        if col in events_df.columns:
            events_df[col] = pd.to_numeric(events_df[col], errors="coerce")

    # Adjust event onsets for dummy scan removal (off by default — already done upstream)
    if adjust_for_dummy_scans and dummy_scans > 0:
        adjustment = dummy_scans * tr
        logger.info("Adjusting onsets by -%.2fs for dummy scan removal", adjustment)
        events_df["onset"] -= adjustment
        events_df = events_df[events_df["onset"] >= 0].copy()

    # Drop events whose onset is past the BOLD's wall time (salvaged scans).
    if n_scans is not None and "onset" in events_df.columns:
        bold_duration = n_scans * tr
        before = len(events_df)
        events_df = events_df[events_df["onset"] < bold_duration].copy()
        dropped = before - len(events_df)
        if dropped > 0:
            logger.info(
                "Dropped %d event(s) with onset >= BOLD duration (%.2fs)",
                dropped,
                bold_duration,
            )

    # Add constant column for modeling
    events_df["constant_1_column"] = 1

    # Initialize junk column if it doesn't exist
    if "junk" not in events_df.columns:
        events_df["junk"] = 0

    # Handle negative RTs: mark as junk and set to NaN
    if "response_time" in events_df.columns:
        na_mask = events_df["response_time"] < 0
        events_df["na_trials"] = na_mask.astype(int)
        events_df.loc[na_mask, "junk"] = 1
        events_df.loc[na_mask, "response_time"] = np.nan
    else:
        events_df["na_trials"] = 0

    return events_df


def define_nuisance_trials(events_df: pd.DataFrame, task: str) -> dict[str, pd.Series]:
    """Define nuisance trials based on task type and response patterns.

    Args:
        events_df: Events dataframe with trial data
        task: Task name

    Returns:
        Dictionary with keys: 'trial_filter', 'bad_trials', 'omission',
        'commission', 'rt_too_fast'.  Each value is a boolean Series mask.
    """
    # Base + dual tasks whose nuisance trials are defined on the test_trial row.
    test_trial_tasks = {
        "cuedTS",
        "nBack",
        "spatialTS",
        "flanker",
        "shapeMatching",
        "directedForgetting",
        # dual tasks (9 non-stop):
        "cuedTSWFlanker",
        "directedForgettingWCuedTS",
        "directedForgettingWFlanker",
        "flankerWShapeMatching",
        "nBackWShapeMatching",
        "nBackWSpatialTS",
        "shapeMatchingWCuedTS",
        "spatialTSWCuedTS",
        "spatialTSWShapeMatching",
    }
    # Base stop/go tasks use bare trial_type == 'go'; stop duals encode go as
    # 'go_congruent'/'go_con'/etc., so match the prefix. Restricting to go trials
    # keeps successful stops (key_press == -1) from being mis-flagged as omissions.
    go_trial_tasks = {"stopSignal", "goNogo"}
    stop_dual_tasks = {"stopSignalWFlanker", "stopSignalWDirectedForgetting"}

    if task in test_trial_tasks:
        trial_filter = events_df.trial_id == "test_trial"
    elif task in go_trial_tasks:
        trial_filter = events_df.trial_type == "go"
    elif task in stop_dual_tasks:
        trial_filter = events_df.trial_type.astype(str).str.startswith("go")
    else:
        raise ValueError(
            f"Unknown task: {task}. Supported tasks: "
            f"{test_trial_tasks | go_trial_tasks | stop_dual_tasks}"
        )

    # Define nuisance trial types
    omission = (events_df.key_press == -1) & trial_filter
    commission = (
        (events_df.key_press != events_df.correct_response)
        & (events_df.key_press != -1)
        & (events_df.response_time >= MIN_RT)
        & trial_filter
    )
    rt_too_fast = (events_df.response_time < MIN_RT) & trial_filter

    # Also include trials already marked as junk
    existing_junk = pd.Series(False, index=events_df.index)
    if "junk" in events_df.columns:
        existing_junk = (events_df["junk"] == 1) & trial_filter

    bad_trials = omission | commission | rt_too_fast | existing_junk

    return {
        "trial_filter": trial_filter,
        "bad_trials": bad_trials,
        "omission": omission,
        "commission": commission,
        "rt_too_fast": rt_too_fast,
    }


def add_junk_trials(
    events_df: pd.DataFrame, task_name: str
) -> tuple[pd.DataFrame, float]:
    """Calculate percentage of junk trials and add nuisance regressors to dataframe.

    Args:
        events_df: Preprocessed events dataframe
        task_name: Name of the task

    Returns:
        Tuple of (events_df with nuisance columns, percentage of junk trials (0-1)).
    """
    if len(events_df) == 0:
        raise ValueError("Events dataframe is empty")

    events_df = events_df.copy()

    # Get nuisance trial masks
    nuisance_masks = define_nuisance_trials(events_df, task_name)

    # Add nuisance columns to dataframe as integers (0/1)
    events_df["junk_trials"] = nuisance_masks["bad_trials"].astype(int)
    events_df["omission"] = nuisance_masks["omission"].astype(int)
    events_df["commission"] = nuisance_masks["commission"].astype(int)
    events_df["rt_too_fast"] = nuisance_masks["rt_too_fast"].astype(int)

    # Denominator is the number of relevant trials (test/go), not all events,
    # so that non-test events (breaks, cues, etc.) don't dilute the junk rate.
    n_relevant = nuisance_masks["trial_filter"].sum()
    junk_percentage = (
        nuisance_masks["bad_trials"].sum() / n_relevant if n_relevant > 0 else 0.0
    )

    return events_df, junk_percentage

def add_ssd_regressor(
        events: pd.DataFrame,
        ssd_col: str = "SS_delay",
        ssd_all: str = "ssd_all",
        ssd_sf: str = "ssd_stop_fail",
) -> pd.DataFrame:
    """
    Adds two per-run ssd regressors:
    - output_all: demeaned SSD on all stop trials
    - output_sf: demeaned SSD on stop_failure trials only
    """
    df = events.copy()

    if ssd_col not in df.columns:
        logger.warning(
            "ssd_regressor: no SS_delay column '%s' found; creating '%s' and '%s' as all-NaN",
            ssd_col, ssd_all, ssd_sf,
        )
        df[ssd_all] = np.nan
        df[ssd_sf] = np.nan

        return df

    df["ssd_num"] = pd.to_numeric(df[ssd_col], errors = "coerce")  #numeric SSD

    if "trial_type" in df.columns:
        # stop success and stop failure
        all_mask = df["trial_type"].astype(str).str.startswith("stop")
        # just stop failures taking into account the stop dual tasks
        sf_mask = df["trial_type"].astype(str).str.startswith("stop_failure")
    else:
        logger.warning("ssd_regressor: no 'trial_type' column found; not adding SSD regressors")
        df[ssd_all] = np.nan
        df[ssd_sf] = np.nan
        df.drop(columns=["ssd_num"], inplace=True)
        return df

    #ssd_all
    ssd_all_vals = df.loc[all_mask, "ssd_num"]
    if ssd_all_vals.notna().sum() == 1:
        logger.warning(
            "output_all: run produced one defined ssd, after centering it would be 0.0 and dropped downstream. Leaving all '%s' NaN",
            ssd_all,
        )

    if ssd_all_vals.notna().sum() > 1:
        mean_all = float(ssd_all_vals.mean(skipna=True))
        df[ssd_all] = np.where(all_mask, df["ssd_num"] - mean_all, np.nan)
    else:
        df[ssd_all] = np.nan
        logger.debug("ssd_regressor: no numeric SS_delay values on stop trials; '%s' all NaN", ssd_all)

    #ssd_sf
    ssd_sf_vals = df.loc[sf_mask, "ssd_num"]
    if ssd_sf_vals.notna().sum() == 1:    #
        logger.warning(
            "ssd_sf: run produced one defined ssd, after centering it would be 0.0 and dropped downstream. Leaving all '%s' NaN",
            ssd_sf,
        )

    if ssd_sf_vals.notna().sum() > 1:
        mean_sf = ssd_sf_vals.mean(skipna=True)
        df[ssd_sf] = np.where(sf_mask, df["ssd_num"] - mean_sf, np.nan)
    else:
        df[ssd_sf] = np.nan
        logger.debug("ssd_regressor: no numeric SS_delay values on stop_failure trials; '%s' all NaN", ssd_sf)

    df.drop(columns = ["ssd_num"], inplace=True)

    return df


def stop_fail_violation(
    events: pd.DataFrame,
    trial_id: str = "trial_id",
    trial_type: str = "trial_type",
    rt: str = "response_time",
    output: str = "stop_failure_violation",
    go: str = "go",
) -> pd.DataFrame:
    """
    Add a per-run column with the violation amplitude for stop_failure trials
    For each stop_failure trial:
      amplitude = stop_failure_rt of trial N - valid_go_rt of trial N-1

    Note: If a run produces 0 or 1 defined violations, no regressor is produced
    (the column is left all NaN) and the corresponding contrast is dropped at the
    design-matrix stage (handle_zero_variance_columns/filter_contrasts_for_dropped_columns)
    -- a run with too few violations silently loses this contrast instead of erroring.

    Scope note: task-name gating for this regressor (in runner.py) currently
    matches only "stopsignal"/"stop_signal" and does not extend to the dual
    tasks (e.g. stopSignalWFlanker).
    """
    df = events.copy()

    df["rt_num"] = pd.to_numeric(df[rt], errors="coerce")
    order = list(df.sort_values("onset"))

    # result Series indexed like the original and filled with NaN
    result = pd.Series(np.nan, index=df.index, dtype=float)

    def check_if_test_trial(label) -> bool:
        # Returns true if the row is a test_trial and false if it is a test_fixation or na/n
        if label is None or label not in df.index:
            return False
        val = df.at[label, trial_id]
        return val == "test_trial"

    def prev_is_valid_go(prev_label) -> bool:
        if prev_label is None or prev_label not in df.index:
            return False

        if df.at[prev_label, trial_type] != go:  # check if prev test_trial is go
            return False

        prev_rt = df.at[prev_label, "rt_num"]  # RT invalid if -1 or NaN
        if pd.isna(prev_rt):
            return False
        if prev_rt == -1:  # omission
            return False
        if prev_rt < MIN_RT:  # too fast
            return False

        if ("key_press" in df.columns) and (
            "correct_response" in df.columns
        ):  # checks go_acc
            key_press = pd.to_numeric(
                df.at[prev_label, "key_press"], errors="coerce"
            )
            correct_resp = pd.to_numeric(
                df.at[prev_label, "correct_response"], errors="coerce"
            )
            if key_press != correct_resp:
                return False

        return True

    amplitudes = []
    amp_labels = []
    prev_test_label = None

    for label in order:
        if trial_id and trial_id in df.columns:
            if not check_if_test_trial(label):  # skips non_test trials
                continue
        else:
            logger.warning(
                "no valid trial_id found; treating all rows as non_test (no violations will be found)"
            )
            continue

        cur_trial_type = df.at[label, trial_type]

        if cur_trial_type == "stop_failure":
            if prev_test_label is None:  # no previous test trial to pair with
                prev_test_label = label
                continue

            cur_rt = df.at[label, "rt_num"]
            if pd.isna(cur_rt):
                prev_test_label = label
                continue
            if cur_rt == -1:
                prev_test_label = label
                continue
            if cur_rt < MIN_RT:
                prev_test_label = label
                continue

            if prev_is_valid_go(prev_test_label):
                prev_rt_val = df.at[prev_test_label, "rt_num"]
                amp = float(cur_rt) - float(prev_rt_val)
                amplitudes.append(amp)
                amp_labels.append(label)
        prev_test_label = label

    n_violations = len(amplitudes)
    if n_violations == 0:
        logger.info(
            "stop_fail_violation: run produced zero defined violations; '%s' is all NaN",
            output,
        )
    elif n_violations == 1:
        logger.warning(
            "stop_fail_violation: run produced one defined violation, after centering it would be 0.0 and dropped downstream. Leaving all '%s' NaN",
            output,
        )
    else:
        mean_amp = float(np.mean(amplitudes))
        centered = [each - mean_amp for each in amplitudes]
        for label, v in zip(amp_labels, centered):
            result.at[label] = v

    df[output] = result
    df = df.drop(columns=["rt_num"])
    return df


def save_simplified_events(regressor_3cols: list, output_file: str | Path) -> Path:
    """Save simplified events in 3-column format.

    Args:
        regressor_3cols: List of (3col_tuple, name) pairs from create_design_matrix
        output_file: Path to save simplified events CSV

    Returns:
        Path to saved file.
    """
    output_file = Path(output_file)

    if not regressor_3cols:
        raise ValueError("No regressors provided - regressor_3cols is empty")

    # Convert 3-column tuples to dataframes
    all_events = []
    for (onsets, durations, amplitudes), regressor_name in regressor_3cols:
        if onsets:  # Only if regressor has events
            regressor_df = pd.DataFrame(
                {
                    "onset": onsets,
                    "duration": durations,
                    "amplitude": amplitudes,
                    "regressor": regressor_name,
                }
            )
            # Filter out zero-amplitude entries to avoid redundant rows
            regressor_df = regressor_df[regressor_df["amplitude"] != 0.0]
            if not regressor_df.empty:
                all_events.append(regressor_df)

    # Combine all regressors
    if not all_events:
        raise ValueError("No valid events found after processing regressors")

    simplified_df = pd.concat(all_events, ignore_index=True)

    # Sort by onset time
    simplified_df = simplified_df.sort_values("onset").reset_index(drop=True)

    # Save to CSV
    simplified_df.to_csv(output_file, index=False)

    return output_file


def load_bold_data_with_dummy_removal(bold_file: str | Path, dummy_scans: int = 0):
    """Load BOLD data and optionally remove dummy scans.

    Default is 0 since BOLD is pre-trimmed by scripts/trim_bold.py in this
    project's workflow. Pass dummy_scans > 0 only if the input BOLD has not
    been trimmed.

    Args:
        bold_file: Path to 4D BOLD NIfTI file
        dummy_scans: Number of dummy scans to remove (default 0)

    Returns:
        BOLD image with dummy scans removed (unchanged if dummy_scans=0).
    """
    from nilearn.image import load_img

    img = load_img(bold_file)

    if dummy_scans > 0:
        return img.slicer[:, :, :, dummy_scans:]
    return img
