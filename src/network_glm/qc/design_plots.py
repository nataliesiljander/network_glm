"""Render the design matrix, contrast matrix, and regressor correlation matrix.

These are the figures people ask for when reviewing a first-level model: "show me
the design", and "show me how collinear the regressors are". Both are produced from
``*_desc-designMatrix.csv``, which
:func:`network_glm.lev1.processing.quality_control.run_quality_control` already
writes for every run — so this is a post-hoc pass over existing lev1 output, not a
rerun. Same principle as ``--skip-qc-plots``: persist the data, render offline.

Three figures per run:

``*_desc-designMatrix.png``
    scans x regressors, via nilearn's ``plot_design_matrix``.
``*_desc-contrastMatrix.png``
    contrasts x regressors. Contrasts come from the task config when not passed
    explicitly, so this works on output written long before the plots existed.
``*_desc-designCorrelation.png``
    regressor x regressor correlation.

We compute the correlation ourselves rather than calling nilearn's
``plot_design_matrix_correlation``, which documents that "the drift and constant
regressors are omitted from the plot". Those are precisely the rows you want when
checking whether a task regressor is soaking up low-frequency drift, so they stay
(``--omit-drift`` opts into nilearn's convention).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs on compute nodes and in CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from nilearn.glm.contrasts import expression_to_contrast_vector  # noqa: E402
from nilearn.plotting import plot_design_matrix  # noqa: E402

logger = logging.getLogger(__name__)

DESIGN_GLOB = "*_desc-designMatrix.csv"

# Columns nilearn's own correlation plot drops. We keep them by default but need to
# recognise them for --omit-drift.
_DRIFT_PREFIXES = ("cosine", "drift")
_CONSTANT_NAMES = ("constant", "intercept")

_ENTITY_RE = re.compile(
    r"(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_task-(?P<task>[^_]+)_(?P<run>run-[^_]+)"
)


def load_design(csv_path) -> pd.DataFrame:
    """Read a persisted design matrix. Written with ``index=False``, so rows are
    scans in acquisition order and there is no index column to drop."""
    return pd.read_csv(Path(csv_path))


def parse_entities(filename: str) -> dict[str, str]:
    """BIDS entities from a design-matrix filename, or ``{}`` if it isn't one."""
    m = _ENTITY_RE.search(str(filename))
    return m.groupdict() if m else {}


def figure_title(entities: dict[str, str], kind: str) -> str:
    """Self-identifying title — these figures get shared without their filename."""
    who = " ".join(
        entities.get(k, "") for k in ("subject", "session", "run") if entities.get(k)
    )
    task = entities.get("task", "")
    label = {
        "design": "design",
        "contrasts": "contrasts",
        "correlation": "regressor correlation",
    }.get(kind, kind)
    return f"{task} {label} — {who}".strip()


def _is_drift_or_constant(column: str) -> bool:
    lower = column.lower()
    return lower.startswith(_DRIFT_PREFIXES) or lower in _CONSTANT_NAMES


def design_correlation(design: pd.DataFrame, *, include_drift: bool = True) -> pd.DataFrame:
    """Regressor x regressor correlation matrix.

    ``include_drift=False`` reproduces nilearn's convention of dropping drift and
    constant regressors. A constant column has zero variance, so its off-diagonal
    correlations are undefined: those stay NaN (rendered blank) while the diagonal
    is forced to 1.0, so the plot doesn't have a hole punched in it.
    """
    if not include_drift:
        keep = [c for c in design.columns if not _is_drift_or_constant(c)]
        design = design[keep]
    corr = design.corr()
    # pandas 3 hands back a read-only view from `.values`, so fill on an explicit copy.
    arr = corr.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, 1.0)
    return pd.DataFrame(arr, index=corr.index, columns=corr.columns)


def contrast_matrix(
    design_columns: list[str], contrasts: dict[str, str]
) -> pd.DataFrame:
    """Contrasts (rows) x regressors (columns), from the same expression parser lev1 uses.

    A contrast naming a regressor this run doesn't have is dropped with a warning
    rather than raising: runs legitimately lose columns (see
    ``contrasts.filter_contrasts_for_dropped_columns``), and a partial figure beats
    no figure.
    """
    rows, names = [], []
    for name, expression in contrasts.items():
        try:
            rows.append(np.asarray(expression_to_contrast_vector(expression, design_columns)))
            names.append(name)
        except Exception as exc:  # noqa: BLE001 — nilearn raises several types here
            logger.warning("skipping contrast %r (%s): %s", name, expression, exc)
    if not rows:
        return pd.DataFrame(columns=design_columns)
    return pd.DataFrame(np.vstack(rows), index=names, columns=design_columns)


def _resolve_contrasts(entities: dict[str, str]) -> dict[str, str]:
    """Look up the task's contrasts, tolerating a task the config doesn't know."""
    task = entities.get("task")
    if not task:
        return {}
    try:
        from network_glm.task_config.loader import get_task_contrasts

        return get_task_contrasts(task)
    except Exception as exc:  # noqa: BLE001 — unknown task / malformed yaml
        logger.warning("no contrasts for task %r: %s", task, exc)
        return {}


def plot_design_png(design: pd.DataFrame, out_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(design.columns)), 9))
    plot_design_matrix(design, axes=ax)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel("scan number")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _plot_heatmap(
    frame: pd.DataFrame, out_path: Path, title: str, *, cmap: str, vmax: float
) -> Path:
    """Shared renderer for the contrast and correlation panels (both are labelled,
    diverging, zero-centred matrices)."""
    n_rows, n_cols = frame.shape
    fig, ax = plt.subplots(
        figsize=(max(7, 0.30 * n_cols), max(5, 0.30 * n_rows + 2))
    )
    im = ax.imshow(
        frame.to_numpy(dtype=float), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto"
    )
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(frame.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(frame.index, fontsize=7)
    ax.set_title(title, fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_run(
    csv_path,
    out_dir=None,
    *,
    contrasts: dict[str, str] | None = None,
    include_drift: bool = True,
) -> dict[str, Path]:
    """Render every figure for one run. Returns ``{kind: path}``.

    ``out_dir`` defaults to the CSV's own directory, so pointing this at a lev1 QC
    directory leaves the figures next to the data they describe. The contrast panel
    is omitted (not failed) when no contrast resolves.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir) if out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    design = load_design(csv_path)
    entities = parse_entities(csv_path.name)
    stem = csv_path.name.replace("_desc-designMatrix.csv", "")
    written: dict[str, Path] = {}

    written["design"] = plot_design_png(
        design,
        out_dir / f"{stem}_desc-designMatrix.png",
        figure_title(entities, "design"),
    )

    corr = design_correlation(design, include_drift=include_drift)
    written["correlation"] = _plot_heatmap(
        corr,
        out_dir / f"{stem}_desc-designCorrelation.png",
        figure_title(entities, "correlation"),
        cmap="RdBu_r",
        vmax=1.0,
    )

    resolved = contrasts if contrasts is not None else _resolve_contrasts(entities)
    if resolved:
        cmat = contrast_matrix(design.columns.tolist(), resolved)
        if not cmat.empty:
            written["contrasts"] = _plot_heatmap(
                cmat,
                out_dir / f"{stem}_desc-contrastMatrix.png",
                figure_title(entities, "contrasts"),
                cmap="RdBu_r",
                vmax=float(np.abs(cmat.to_numpy()).max() or 1.0),
            )
    if "contrasts" not in written:
        logger.warning("no contrasts rendered for %s", csv_path.name)

    return written


def _collect_inputs(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(target.rglob(DESIGN_GLOB))
    return [target] if target.is_file() else []


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="network-glm design-plots",
        description=(
            "Plot the design matrix, contrast matrix, and regressor correlation "
            "matrix from persisted lev1 design matrices."
        ),
    )
    parser.add_argument(
        "target",
        help=f"a {DESIGN_GLOB} file, or a directory to search recursively",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="where to write the figures (default: beside each input CSV)",
    )
    parser.add_argument(
        "--omit-drift",
        action="store_true",
        help="drop drift/constant regressors from the correlation plot "
        "(nilearn's convention; they are kept by default)",
    )
    return parser


def main(argv=None) -> int:
    args = get_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    target = Path(args.target)
    inputs = _collect_inputs(target)
    if not inputs:
        raise SystemExit(f"no {DESIGN_GLOB} found at {target}")

    total = 0
    for csv_path in inputs:
        written = plot_run(
            csv_path, out_dir=args.out_dir, include_drift=not args.omit_drift
        )
        total += len(written)
        logger.info("%s -> %s", csv_path.name, ", ".join(sorted(written)))
    logger.info("wrote %d figure(s) for %d run(s)", total, len(inputs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
