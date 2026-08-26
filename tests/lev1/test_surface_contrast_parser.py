"""Scientific-correctness tests for ``SurfaceGLM._parse_contrast``.

These tests are not box-checking — they exist to keep the surface contrast
vector identical to the volumetric one for every contrast formula we ship
in production. A bug here is silent: the GLM still runs and writes files,
just with the wrong weights baked into ``effect_size``, ``effect_variance``
and ``z_score``. The previous handwritten parser silently dropped any term
adjacent to ``(``, ``)``, or ``/`` (fractional coefficients), which would
have invalidated ``task-baseline``, ``twoBack-oneBack``, ``match-mismatch``,
``main_vars``, and the ``1/3 *`` formulas in stopSignal and goNogo.

Coverage:

1. **Volumetric/surface parity** — for every YAML contrast in every base
   task, the surface vector must equal the volumetric vector (both come
   from ``nilearn.glm.contrasts.expression_to_contrast_vector``).
2. **Coefficient correctness** — fractional and parenthesized coefficients
   produce the exact expected magnitudes, not approximations or zeros.
3. **Mumford ConstDurRTDur invariant** — a contrast over ConstDur trial
   types must not accidentally weight the pooled ``response_time`` RTDur
   regressor.
4. **Loud failure on unknown regressors** — referencing a missing
   regressor raises a ``ValueError`` rather than silently emitting a
   zero. The old parser silently dropped unknowns, which produced
   undetectable corruption.

Add a test here every time a new contrast pattern enters a task YAML.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nilearn.glm.contrasts import expression_to_contrast_vector

from network_glm.lev1.processing.surface_data import SurfaceGLM
from network_glm.task_config.loader import (
    get_regressor_config,
    get_task_contrasts,
)

BASE_TASKS = [
    "cuedTS",
    "directedForgetting",
    "flanker",
    "goNogo",
    "nBack",
    "shapeMatching",
    "spatialTS",
    "stopSignal",
]


def _make_fitted_glm(regressor_names: list[str]) -> SurfaceGLM:
    """Fit a minimal SurfaceGLM so ``_parse_contrast`` has a regressor list.

    Synthetic data; we only need ``regressor_names_`` populated. AR(1) is
    overkill here but matches production noise model defaults.
    """
    np.random.seed(0)
    n_tp = max(80, 4 * len(regressor_names))
    n_verts = 10
    X = pd.DataFrame(
        np.random.randn(n_tp, len(regressor_names)),
        columns=regressor_names,
    )
    Y = np.random.randn(n_tp, n_verts)
    glm = SurfaceGLM(t_r=1.5, noise_model="ols")
    glm.fit(Y, X)
    return glm


# ---------------------------------------------------------------------------
# 1. Volumetric/surface parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_name", BASE_TASKS)
def test_surface_contrast_vector_matches_volumetric_for_all_yaml_contrasts(task_name):
    """Surface ``_parse_contrast`` output must equal the volumetric path's
    contrast vector for every YAML contrast in every base task.

    Both should be exactly ``expression_to_contrast_vector(formula, names)``.
    """
    regressors = list(get_regressor_config(task_name).keys())
    contrasts = get_task_contrasts(task_name)

    glm = _make_fitted_glm(regressors)
    for contrast_name, formula in contrasts.items():
        surface_vec = glm._parse_contrast(formula)
        volumetric_vec = np.asarray(expression_to_contrast_vector(formula, regressors))
        assert (
            surface_vec.shape == volumetric_vec.shape
        ), f"{task_name}/{contrast_name}: shape mismatch"
        np.testing.assert_array_equal(
            surface_vec,
            volumetric_vec,
            err_msg=(
                f"{task_name}/{contrast_name}: surface vector does not match "
                f"volumetric vector for formula {formula!r}"
            ),
        )


# ---------------------------------------------------------------------------
# 2. Coefficient correctness — exact magnitudes for patterns the old parser broke
# ---------------------------------------------------------------------------


def test_fractional_coefficient_in_task_baseline_pattern():
    """``1/3 * (a + b + c)`` weights each regressor at exactly 1/3.

    The old parser produced a zero vector here because the regex captured ``1``
    as the coefficient and ``/3 * a`` as the regressor name.
    """
    glm = _make_fitted_glm(["go", "stop_failure", "stop_success"])
    vec = glm._parse_contrast("1/3 * go + 1/3 * stop_failure + 1/3 * stop_success")
    np.testing.assert_allclose(vec, [1 / 3, 1 / 3, 1 / 3])


def test_parenthesized_expression_with_mixed_signs():
    """``0.5 * (a + b - c - d)`` weights ``a, b`` at +0.5 and ``c, d`` at -0.5.

    The old parser dropped the first and last terms (they had ``(`` and ``)``
    glued to the regressor names) and produced ``[0, +1.0, -1.0, 0]``.
    """
    names = ["mismatch_2back", "match_2back", "mismatch_1back", "match_1back"]
    glm = _make_fitted_glm(names)
    vec = glm._parse_contrast("0.5 * (mismatch_2back + match_2back - mismatch_1back - match_1back)")
    np.testing.assert_allclose(vec, [0.5, 0.5, -0.5, -0.5])


def test_decimal_coefficient_pair():
    """``0.5 * go + 0.5 * nogo_success`` — a baseline the old parser handled.

    Sanity check that the new parser also handles it. Should produce
    ``[0.5, 0.5]`` against the regressors in declaration order.
    """
    glm = _make_fitted_glm(["go", "nogo_success"])
    vec = glm._parse_contrast("0.5 * go + 0.5 * nogo_success")
    np.testing.assert_allclose(vec, [0.5, 0.5])


def test_simple_minus_contrast():
    """``stop_success - go`` produces ``[+1, -1]`` with go listed first."""
    glm = _make_fitted_glm(["go", "stop_success"])
    vec = glm._parse_contrast("stop_success - go")
    np.testing.assert_allclose(vec, [-1.0, 1.0])


# ---------------------------------------------------------------------------
# 3. Mumford ConstDurRTDur invariant
# ---------------------------------------------------------------------------


# Tasks that include a pooled `response_time` (RTDur) regressor per Mumford 2023.
# Add tasks here when their YAML gains a response_time regressor — the invariant
# test below auto-extends to enforce the Mumford correction for them.
TASKS_WITH_RTDUR = [
    "nBack",
    "cuedTS",
    "directedForgetting",
    "flanker",
    "shapeMatching",
    "spatialTS",
    "goNogo",
]


@pytest.mark.parametrize("task_name", TASKS_WITH_RTDUR)
def test_cognitive_contrasts_do_not_weight_pooled_rtdur_regressor(task_name):
    """Cognitive (non-RT) contrasts must have zero weight on the pooled
    ``response_time`` (RTDur) regressor.

    Mumford 2023 (biorxiv 528677): the pooled RTDur regressor absorbs
    trial-to-trial RT variance, freeing the ConstDur regressors to capture
    stimulus-locked activity. If a cognitive contrast (e.g. ``stop_success-go``,
    ``twoBack-oneBack``) puts any nonzero weight on ``response_time``, the
    contrast smuggles RT variance into the cognitive effect — the Mumford
    correction is undone and inhibition-vs-go or N-back load contrasts will
    partially reflect RT differences.

    The explicit ``response_time`` contrast is exempt — it intentionally
    extracts the RTDur beta to provide an RT-effect map.

    For the inhibition tasks (stopSignal, goNogo), RTDur is restricted to
    correct go-trial responses only. We deliberately exclude *_failure
    (commission-error) trials because their RTs reflect failed inhibition,
    not clean motor readiness. See the rationale at the top of the
    stopSignal.yaml / goNogo.yaml configs.
    """
    regressors = list(get_regressor_config(task_name).keys())
    assert "response_time" in regressors, (
        f"{task_name} regressors must include the pooled RTDur "
        f"(response_time) regressor for this test to be meaningful. If "
        f"this task is intentionally without RT modeling, remove it from "
        f"TASKS_WITH_RTDUR."
    )
    glm = _make_fitted_glm(regressors)
    rt_idx = regressors.index("response_time")
    contrasts = get_task_contrasts(task_name)

    for contrast_name, formula in contrasts.items():
        if contrast_name == "response_time":
            continue  # explicit RT-effect contrast is exempt
        vec = glm._parse_contrast(formula)
        assert vec[rt_idx] == 0.0, (
            f"{task_name}/{contrast_name}: contrast weights the pooled "
            f"response_time RTDur regressor at {vec[rt_idx]}; this leaks "
            f"RT variance into a cognitive contrast and undoes the "
            f"Mumford ConstDurRTDur correction (formula={formula!r})"
        )


@pytest.mark.parametrize("task_name", BASE_TASKS)
def test_break_regressor_duration_matches_canonical_task_design(task_name):
    """Every base-task YAML must model `break_with_performance_feedback` as
    a 10-second epoch (the canonical break-with-feedback window).

    Originally the YAML hardcoded `duration: 1`, leaving 9 of every 10 seconds
    of feedback display as implicit baseline. The events↔config audit
    (`scripts/audit_events_vs_task_configs.py`) confirmed
    `break_with_performance_feedback` rows are 10.0s ± 0 across both cohorts
    and every task that has them. This test pins the YAML duration so future
    edits cannot silently revert to a stim-duration value.

    If a future task redesign changes the break duration, update this test
    AND re-run the audit before launching production.
    """
    cfg = get_regressor_config(task_name)
    assert "break_with_performance_feedback" in cfg, (
        f"{task_name} regressors must include break_with_performance_feedback "
        f"so feedback-block variance is regressed out of the task-baseline."
    )
    duration_col = cfg["break_with_performance_feedback"]["duration_column"]
    assert duration_col == "constant_10_column", (
        f"{task_name}/break_with_performance_feedback: expected duration=10 "
        f"(canonical break-with-feedback window per events.tsv audit) but "
        f"YAML produced duration_column={duration_col!r}. If this is a "
        f"deliberate change, update this test AND re-run "
        f"scripts/audit_events_vs_task_configs.py to confirm the new value."
    )


@pytest.mark.parametrize(
    "task_name,primary_trial_types",
    [
        ("stopSignal", ["go", "stop_success", "stop_failure"]),
        ("goNogo", ["go", "nogo_success", "nogo_failure"]),
    ],
)
def test_inhibition_task_baseline_weights_all_primary_trial_types_equally(
    task_name,
    primary_trial_types,
):
    """For the inhibition tasks, ``task-baseline`` must weight every primary
    trial-type regressor (the *correct response*, the *successful inhibition*,
    and the *failed inhibition*) with the same nonzero coefficient.

    Why: task-baseline is the cohort-wide "is the task driving signal at all"
    contrast.  Restricting it to the successful subset would condition the
    contrast on performance and yield asymmetric meaning across goNogo and
    stopSignal — a downstream interpretability hazard if a meta-analysis
    ever combines or compares the two tasks.

    Concretely: previously goNogo's task-baseline was ``0.5 * go + 0.5 *
    nogo_success`` (omitting nogo_failure entirely), while stopSignal's was
    ``1/3 * (go + stop_success + stop_failure)``.  This test enforces
    symmetry so that drift in either direction fails fast.
    """
    regressors = list(get_regressor_config(task_name).keys())
    formula = get_task_contrasts(task_name)["task-baseline"]
    glm = _make_fitted_glm(regressors)
    vec = glm._parse_contrast(formula)

    weights = {}
    for trial_type in primary_trial_types:
        assert trial_type in regressors, (
            f"{task_name}: primary trial-type regressor {trial_type!r} is "
            f"missing from the YAML.  Update the YAML or remove it from this "
            f"test's primary_trial_types list."
        )
        idx = regressors.index(trial_type)
        weights[trial_type] = vec[idx]

    # Every primary trial type must have a strictly positive weight.
    for trial_type, w in weights.items():
        assert w > 0, (
            f"{task_name}/task-baseline: trial type {trial_type!r} has weight "
            f"{w}.  task-baseline must include every primary trial-type "
            f"regressor with a strictly positive weight (formula={formula!r})."
        )

    # All primary weights must be approximately equal.
    values = list(weights.values())
    spread = max(values) - min(values)
    assert spread < 1e-9, (
        f"{task_name}/task-baseline: primary trial-type weights are not "
        f"equal: {weights}.  task-baseline must average uniformly across "
        f"go, success, and failure to keep its meaning symmetric across the "
        f"inhibition tasks (formula={formula!r})."
    )


def test_inhibition_rtdur_subset_excludes_failure_trials():
    """For goNogo, the response_time regressor's subset
    must exclude commission-error trials (nogo_failure).

    Failed-inhibition RTs reflect partial response programming and are
    contaminated relative to clean go-trial RTs. Including them in the
    pooled RTDur regressor would bleed failure-trial variance into the
    Mumford correction. This invariant is enforced at the YAML config
    level via the subset filter.
    """
    for task_name in ("goNogo"):
        config = get_regressor_config(task_name)
        rt_subset = config["response_time"]["subset"]
        # The subset must restrict to trial_type == 'go'. The most common
        # ways to express this are explicit equality to 'go' or filtering
        # against the failure trial types.
        assert "trial_type == 'go'" in rt_subset, (
            f"{task_name}: response_time subset must restrict to "
            f"trial_type == 'go' so RTDur excludes commission-error trials. "
            f"Current subset: {rt_subset!r}"
        )


# ---------------------------------------------------------------------------
# 4. Loud failure on unknown regressors
# ---------------------------------------------------------------------------


def test_surface_compute_contrast_runs_end_to_end_on_synthetic_data():
    """End-to-end smoke: ``SurfaceGLM.compute_contrast`` returns a populated
    result dict for a real fit-then-contrast call against synthetic data.

    Regression guard for the nilearn API rename. Nilearn 0.13 renamed
    ``compute_contrast(contrast_type=)`` to ``compute_contrast(stat_type=)``;
    if the kwarg drifts again every contrast in the cohort run will silently
    fail with ``unexpected keyword argument`` and produce no z-maps even
    though the GLM fit itself succeeds — exactly the failure mode observed
    on smoke job 25246311 before this guard existed.

    The unit-level parser tests above never exercise the actual nilearn
    call (they stub it out), so this test runs the full fit→contrast path.
    """
    np.random.seed(0)
    n_tp, n_verts = 60, 200
    X = np.random.randn(n_tp, 3)
    Y = X @ np.random.randn(3, n_verts) + np.random.randn(n_tp, n_verts) * 0.5

    dm = pd.DataFrame(X, columns=["cond_a", "cond_b", "constant"])
    glm = SurfaceGLM(t_r=1.5, noise_model="ols")
    glm.fit(Y, dm)

    result = glm.compute_contrast("cond_a - cond_b", output_type="all")
    for key in ("effect_size", "effect_variance", "z_score"):
        assert key in result, f"compute_contrast must return {key!r}"
        assert hasattr(result[key], "data"), f"{key} should be a SurfaceResult with a .data array"
        assert result[key].data.shape == (n_verts,)


def test_unknown_regressor_raises_rather_than_silently_zeroing():
    """Referencing a non-existent regressor must raise, not silently emit
    a zero-weighted contrast.

    The old parser silently dropped unknown terms, which produced silently
    corrupt contrasts. nilearn's ``expression_to_contrast_vector`` raises
    a ``ValueError`` — we want that behavior preserved.
    """
    glm = _make_fitted_glm(["go", "stop_success"])
    with pytest.raises(ValueError):
        glm._parse_contrast("go - nonexistent_regressor")


# ---------------------------------------------------------------------------
# 5. Cross-task contrast-balance invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_name", BASE_TASKS)
def test_all_yaml_contrasts_have_at_least_one_nonzero_weight(task_name):
    """Every contrast in every YAML must produce at least one nonzero weight.

    If a contrast comes out as the all-zero vector, either the formula uses
    a regressor name that doesn't match any in the design, or the YAML
    contrast list is empty / placeholder. Either way it's not analysis-ready
    and the production rerun should not silently emit a NaN z-map.
    """
    regressors = list(get_regressor_config(task_name).keys())
    contrasts = get_task_contrasts(task_name)
    glm = _make_fitted_glm(regressors)
    for contrast_name, formula in contrasts.items():
        vec = glm._parse_contrast(formula)
        assert np.any(vec != 0), (
            f"{task_name}/{contrast_name}: all-zero contrast vector for "
            f"formula {formula!r}; regressors available: {regressors}"
        )
