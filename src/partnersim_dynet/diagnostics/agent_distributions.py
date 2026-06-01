"""Boxplot diagnostic for per-agent effective formation/breakage probabilities.

Produces one figure per (Sex, Orientation) combination — six figures in
total — each showing a 2-row × 6-column grid: rows are Formation /
Breakage, columns are age groups. Each cell is a boxplot of per-agent
effective probabilities for the agents in that combo, with a strip plot
overlay.

The boxplots reveal heterogeneity: a wide box means agents in that
demographic experienced a wide spread of effective probabilities (driven
by the NB multiplier). A narrow box means everyone got similar rates.
This is the diagnostic for "did my NB heterogeneity calibration produce
the dispersion I wanted?".
"""

from __future__ import annotations

import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from partnersim_dynet.config import AGE_GROUPS, PartnershipConfig
from partnersim_dynet.diagnostics.probability_tables import export_probability_bounds
from partnersim_dynet.network.plots.style import OutputFormats, publication_style, save_figure


_SEX_LABEL = {"Male": "Male", "Female": "Female"}


def plot_agent_probability_distributions(
    cfg: PartnershipConfig,
    agent_log: pd.DataFrame,
    output_dir: str,
    filename_prefix: str = "agent_probs",
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """Boxplots of per-agent effective probabilities, by demographic group.

    Produces six figures (Males × Opposite/Same/Bisexual, Females ×
    Opposite/Same/Bisexual). Each figure has a 2×6 grid: rows are
    Formation/Breakage, columns are age groups.

    Parameters
    ----------
    cfg : PartnershipConfig
        Used to recompute effective per-agent probabilities.
    agent_log : DataFrame
        From ``PartnershipGenerator.get_agent_log()``.
    output_dir : str
        Where to save figures.
    filename_prefix : str
        Prefix for output files; full name is
        ``{prefix}_{Sex}_{Orientation}.png``.
    formats : OutputFormats
        Which formats to save.

    Returns
    -------
    list of file paths written.
    """
    # Reuse export_probability_bounds machinery to get per-agent effective probs.
    # We need the per-agent values, not just the per-combo summary, so we
    # do the calculation inline here (mirroring export_probability_bounds).
    from partnersim_dynet.config import age_to_group

    formation = cfg.probabilities.build_formation_probs()
    breakage = cfg.probabilities.build_breakage_probs()

    df = agent_log.copy()
    df["AgeGroup"] = df["EntryAge"].apply(age_to_group)

    def _effective(row, base_table: dict, mult_col: str) -> float:
        base = base_table.get(row["Sex"], {}).get(row["Orientation"], {}).get(row["AgeGroup"], 0.0)
        prob = base * row[mult_col]
        if row["HighActive"]:
            prob *= cfg.high_activity_multiplier
        return max(cfg.prob_floor, min(prob, cfg.prob_ceiling))

    df["Formation"] = df.apply(lambda r: _effective(r, formation, "NBMultiplierForm"), axis=1)
    df["Breakage"] = df.apply(lambda r: _effective(r, breakage, "NBMultiplierBreak"), axis=1)

    sexes = ("Male", "Female")
    orientations = ("Opposite-sex", "Same-sex", "Bisexual")

    # Per (sex, orientation, outcome) y-axis limits — keeps the visual
    # scale meaningful within each figure.
    y_limits: dict = {}
    for sex in sexes:
        y_limits[sex] = {}
        for ori in orientations:
            y_limits[sex][ori] = {}
            for outcome in ("Formation", "Breakage"):
                vals = (
                    df.loc[(df["Sex"] == sex) & (df["Orientation"] == ori), outcome].dropna().values
                )
                if len(vals) == 0:
                    y_limits[sex][ori][outcome] = (0.0, 1.0)
                    continue
                ymin = float(vals.min())
                ymax = float(vals.max())
                pad = 0.05 * (ymax - ymin if ymax > ymin else 1)
                y_limits[sex][ori][outcome] = (ymin - pad, ymax + pad)

    written: list[str] = []

    for sex in sexes:
        for ori in orientations:
            df_sub = df[(df["Sex"] == sex) & (df["Orientation"] == ori)].copy()

            with publication_style():
                fig = plt.figure(figsize=(14, 6))
                gs = gridspec.GridSpec(
                    nrows=2,
                    ncols=6,
                    figure=fig,
                    hspace=0.25,
                    wspace=0.15,
                    left=0.10,
                    right=0.97,
                    top=0.88,
                    bottom=0.12,
                )

                for outcome_idx, outcome in enumerate(("Formation", "Breakage")):
                    for age_idx, age_group in enumerate(AGE_GROUPS):
                        ax = fig.add_subplot(gs[outcome_idx, age_idx])

                        cell = df_sub[df_sub["AgeGroup"] == age_group][[outcome]].rename(
                            columns={outcome: "Probability"}
                        )

                        ax.spines[["top", "right"]].set_visible(False)
                        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
                        ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))
                        ax.yaxis.grid(True, lw=0.4, color="#DDDDDD")
                        ax.set_axisbelow(True)
                        ax.set_ylim(y_limits[sex][ori][outcome])

                        if cell.empty or cell["Probability"].dropna().empty:
                            ax.text(
                                0.5,
                                0.5,
                                "No data",
                                transform=ax.transAxes,
                                ha="center",
                                va="center",
                                fontsize=7,
                                color="grey",
                            )
                            ax.set_xticks([])
                            continue

                        sns.boxplot(
                            data=cell,
                            y="Probability",
                            width=0.6,
                            showfliers=True,
                            ax=ax,
                            color="#D3D3D3",
                            linewidth=0.8,
                        )
                        sns.stripplot(
                            data=cell,
                            y="Probability",
                            color="black",
                            size=2,
                            alpha=0.3,
                            jitter=0.2,
                            ax=ax,
                        )

                        if outcome_idx == 0:
                            ax.set_title(age_group, fontsize=9, fontweight="bold")
                        if age_idx == 0:
                            ax.set_ylabel(outcome, fontsize=9, fontweight="bold")
                        else:
                            ax.set_ylabel("")
                        ax.set_xticks([])
                        ax.set_xlabel("")

                fig.suptitle(
                    f"{_SEX_LABEL[sex]} — {ori} partnerships",
                    fontsize=12,
                    fontweight="bold",
                    y=0.98,
                )
                fig.text(
                    0.02,
                    0.5,
                    "Probability",
                    va="center",
                    ha="center",
                    rotation="vertical",
                    fontsize=10,
                    fontweight="bold",
                )
                fig.text(
                    0.5,
                    0.02,
                    "Age group",
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )

                safe_ori = ori.replace("-", "").replace(" ", "")
                filename = f"{filename_prefix}_{_SEX_LABEL[sex]}_{safe_ori}"
                paths = save_figure(fig, os.path.join(output_dir, filename), formats)
                written.extend(paths)
                plt.close(fig)

    return written
