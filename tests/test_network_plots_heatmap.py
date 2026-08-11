# """Tests for the degree heatmap plot."""

# from __future__ import annotations

# import os

# import matplotlib

# matplotlib.use("Agg")

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import pytest

# from partnersim_dynet.config import AGE_GROUPS
# from partnersim_dynet.network.plots import (
#     OutputFormats,
#     plot_degree_heatmap_evolution,
# )

# # Helpers

# def _make_demographic_df(
#     windows: list[tuple[int, int]],
#     seed: int = 0,
# ) -> pd.DataFrame:
#     """Build a synthetic degree_by_demographic DataFrame covering all windows.

#     Generates one row per (t, AgeGroup, Sex, Orientation) for every
#     timestep that falls inside at least one window.
#     """
#     rng = np.random.default_rng(seed)
#     sexes = ("Male", "Female")
#     oris = ("Opposite-sex", "Same-sex", "Bisexual")

#     # Collect every timestep covered by any window
#     timesteps: list[int] = []
#     for w_start, w_end in windows:
#         timesteps.extend(range(w_start, w_end + 1))
#     timesteps = sorted(set(timesteps))

#     rows = []
#     for t in timesteps:
#         for ag in AGE_GROUPS:
#             for sx in sexes:
#                 for ori in oris:
#                     rows.append(
#                         {
#                             "t": t,
#                             "AgentSex": sx,
#                             "AgentOrientation": ori,
#                             "AgentAgeGroup": ag,
#                             "MeanDegree": float(rng.uniform(0, 2)),
#                             "P50Degree": 0.0,
#                             "P90Degree": 0.0,
#                             "N": rng.integers(5, 50),
#                         }
#                     )
#     return pd.DataFrame(rows)


# # ── Shared window fixtures ────────────────────────────────────────────────────

# # Small windows for unit tests — narrow ranges keep DataFrame size manageable
# WINDOWS_3 = [(51, 55), (56, 60), (61, 65)]
# WINDOWS_2 = [(51, 55), (56, 60)]
# WINDOWS_1 = [(51, 55)]


# # ── plot_degree_heatmap_evolution ─────────────────────────────────────────────


# class TestPlotDegreeHeatmap:
#     def test_writes_png_by_default(self, tmp_path):
#         df = _make_demographic_df(WINDOWS_3)
#         written = plot_degree_heatmap_evolution(
#             df, windows=WINDOWS_3, output_dir=str(tmp_path)
#         )
#         assert len(written) == 1
#         assert written[0].endswith(".png")
#         assert os.path.exists(written[0])
#         assert os.path.getsize(written[0]) > 1000

#     def test_writes_all_formats(self, tmp_path):
#         df = _make_demographic_df(WINDOWS_3)
#         written = plot_degree_heatmap_evolution(
#             df,
#             windows=WINDOWS_3,
#             output_dir=str(tmp_path),
#             formats=OutputFormats.all_enabled(),
#         )
#         assert len(written) == 3

#     def test_custom_filename_stem(self, tmp_path):
#         df = _make_demographic_df(WINDOWS_3)
#         plot_degree_heatmap_evolution(
#             df,
#             windows=WINDOWS_3,
#             output_dir=str(tmp_path),
#             filename_stem="custom_heatmap",
#         )
#         assert os.path.exists(os.path.join(str(tmp_path), "custom_heatmap.png"))

#     def test_empty_windows_raises(self, tmp_path):
#         df = _make_demographic_df(WINDOWS_3)
#         with pytest.raises(ValueError, match="non-empty"):
#             plot_degree_heatmap_evolution(df, windows=[], output_dir=str(tmp_path))

#     def test_missing_column_raises(self, tmp_path):
#         df = pd.DataFrame({"t": [1], "AgentSex": ["Male"]})
#         with pytest.raises(KeyError, match="missing columns"):
#             plot_degree_heatmap_evolution(
#                 df, windows=[(1, 1)], output_dir=str(tmp_path)
#             )

#     def test_single_window_works(self, tmp_path):
#         """When n_cols=1, axes squeeze must not produce a 1-D array."""
#         df = _make_demographic_df(WINDOWS_1)
#         written = plot_degree_heatmap_evolution(
#             df, windows=WINDOWS_1, output_dir=str(tmp_path)
#         )
#         assert os.path.exists(written[0])

#     def test_no_figure_leaks(self, tmp_path):
#         df = _make_demographic_df(WINDOWS_2)
#         before = len(plt.get_fignums())
#         plot_degree_heatmap_evolution(df, windows=WINDOWS_2, output_dir=str(tmp_path))
#         after = len(plt.get_fignums())
#         assert after == before

#     def test_handles_zero_degree_data(self, tmp_path):
#         """All-zero MeanDegree: vmax should fall back to 1.0."""
#         df = _make_demographic_df(WINDOWS_2)
#         df["MeanDegree"] = 0.0
#         written = plot_degree_heatmap_evolution(
#             df, windows=WINDOWS_2, output_dir=str(tmp_path)
#         )
#         assert os.path.exists(written[0])

#     def test_handles_combos_missing_from_some_windows(self, tmp_path):
#         """Missing (t, ori) combos in one window should render as blank cells."""
#         df = _make_demographic_df(WINDOWS_2)
#         # Drop all Same-sex rows from the first window
#         w_start, w_end = WINDOWS_2[0]
#         df = df[
#             ~(
#                 (df["t"] >= w_start)
#                 & (df["t"] <= w_end)
#                 & (df["AgentOrientation"] == "Same-sex")
#             )
#         ]
#         written = plot_degree_heatmap_evolution(
#             df, windows=WINDOWS_2, output_dir=str(tmp_path)
#         )
#         assert os.path.exists(written[0])


# # ── Integration with the full pipeline ───────────────────────────────────────


# class TestIntegrationWithRealMetrics:
#     def test_runs_against_real_metrics_output(self, tmp_path):
#         """End-to-end: simulate, compute degree_by_demographic, plot."""
#         from partnersim_dynet.config import PartnershipConfig
#         from partnersim_dynet.generator import PartnershipGenerator
#         from partnersim_dynet.network import (
#             ActiveIntervals,
#             degree_by_demographic_over_time,
#             prepare_partnerships,
#         )

#         total_timesteps = 200
#         cfg = PartnershipConfig(num_agents=200, total_timesteps=total_timesteps)
#         gen = PartnershipGenerator(cfg, seed=42)
#         df = gen.simulate_partnerships()
#         log = gen.get_agent_log()
#         active = ActiveIntervals.from_agent_log(log, total_timesteps=total_timesteps)
#         arr = prepare_partnerships(df, total_timesteps=total_timesteps)

#         demo = degree_by_demographic_over_time(arr, active, log, total_timesteps=total_timesteps)

#         # 200-step sim: discard first/last 10 as a proportional stand-in
#         # for the 50-step rule, then split remainder into 4 equal windows
#         analysis_start, analysis_end = 11, 190
#         block_width = (analysis_end - analysis_start + 1) // 4  # 45
#         windows = [
#             (analysis_start + i * block_width,
#              analysis_start + (i + 1) * block_width - 1)
#             for i in range(4)
#         ]

#         written = plot_degree_heatmap_evolution(
#             demo,
#             windows=windows,
#             output_dir=str(tmp_path),
#             formats=OutputFormats.all_enabled(),
#         )
#         assert len(written) == 3
#         for p in written:
#             assert os.path.exists(p)
#             assert os.path.getsize(p) > 1000

"""Tests for the degree heatmap plot."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from partnersim_dynet.config import AGE_GROUPS
from partnersim_dynet.network.plots import (
    OutputFormats,
    plot_degree_heatmap_evolution,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_demographic_df(
    windows: list[tuple[int, int]],
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic degree_by_demographic DataFrame covering all windows.

    Generates one row per (t, AgeGroup, Sex, Orientation) for every
    timestep that falls inside at least one window.
    """
    rng = np.random.default_rng(seed)
    sexes = ("Male", "Female")
    oris = ("Opposite-sex", "Same-sex", "Bisexual")

    # Collect every timestep covered by any window
    timesteps: list[int] = []
    for w_start, w_end in windows:
        timesteps.extend(range(w_start, w_end + 1))
    timesteps = sorted(set(timesteps))

    rows = []
    for t in timesteps:
        for ag in AGE_GROUPS:
            for sx in sexes:
                for ori in oris:
                    rows.append(
                        {
                            "t": t,
                            "AgentSex": sx,
                            "AgentOrientation": ori,
                            "AgentAgeGroup": ag,
                            "MeanDegree": float(rng.uniform(0, 2)),
                            "P50Degree": 0.0,
                            "P90Degree": 0.0,
                            "N": rng.integers(5, 50),
                        }
                    )
    return pd.DataFrame(rows)


# ── Shared window fixtures ────────────────────────────────────────────────────

# Small windows for unit tests — narrow ranges keep DataFrame size manageable
WINDOWS_3 = [(51, 55), (56, 60), (61, 65)]
WINDOWS_2 = [(51, 55), (56, 60)]
WINDOWS_1 = [(51, 55)]


# ── plot_degree_heatmap_evolution ─────────────────────────────────────────────


class TestPlotDegreeHeatmap:
    def test_writes_png_by_default(self, tmp_path):
        df = _make_demographic_df(WINDOWS_3)
        written = plot_degree_heatmap_evolution(df, windows=WINDOWS_3, output_dir=str(tmp_path))
        assert len(written) == 1
        assert written[0].endswith(".png")
        assert os.path.exists(written[0])
        assert os.path.getsize(written[0]) > 1000

    def test_writes_all_formats(self, tmp_path):
        df = _make_demographic_df(WINDOWS_3)
        written = plot_degree_heatmap_evolution(
            df,
            windows=WINDOWS_3,
            output_dir=str(tmp_path),
            formats=OutputFormats.all_enabled(),
        )
        assert len(written) == 3

    def test_custom_filename_stem(self, tmp_path):
        df = _make_demographic_df(WINDOWS_3)
        plot_degree_heatmap_evolution(
            df,
            windows=WINDOWS_3,
            output_dir=str(tmp_path),
            filename_stem="custom_heatmap",
        )
        assert os.path.exists(os.path.join(str(tmp_path), "custom_heatmap.png"))

    def test_empty_windows_raises(self, tmp_path):
        df = _make_demographic_df(WINDOWS_3)
        with pytest.raises(ValueError, match="non-empty"):
            plot_degree_heatmap_evolution(df, windows=[], output_dir=str(tmp_path))

    def test_missing_column_raises(self, tmp_path):
        df = pd.DataFrame({"t": [1], "AgentSex": ["Male"]})
        with pytest.raises(KeyError, match="missing columns"):
            plot_degree_heatmap_evolution(df, windows=[(1, 1)], output_dir=str(tmp_path))

    def test_single_window_works(self, tmp_path):
        """When n_cols=1, axes squeeze must not produce a 1-D array."""
        df = _make_demographic_df(WINDOWS_1)
        written = plot_degree_heatmap_evolution(df, windows=WINDOWS_1, output_dir=str(tmp_path))
        assert os.path.exists(written[0])

    def test_no_figure_leaks(self, tmp_path):
        df = _make_demographic_df(WINDOWS_2)
        before = len(plt.get_fignums())
        plot_degree_heatmap_evolution(df, windows=WINDOWS_2, output_dir=str(tmp_path))
        after = len(plt.get_fignums())
        assert after == before

    def test_handles_zero_degree_data(self, tmp_path):
        """All-zero MeanDegree: vmax should fall back to 1.0."""
        df = _make_demographic_df(WINDOWS_2)
        df["MeanDegree"] = 0.0
        written = plot_degree_heatmap_evolution(df, windows=WINDOWS_2, output_dir=str(tmp_path))
        assert os.path.exists(written[0])

    def test_handles_combos_missing_from_some_windows(self, tmp_path):
        """Missing (t, ori) combos in one window should render as blank cells."""
        df = _make_demographic_df(WINDOWS_2)
        # Drop all Same-sex rows from the first window
        w_start, w_end = WINDOWS_2[0]
        df = df[
            ~((df["t"] >= w_start) & (df["t"] <= w_end) & (df["AgentOrientation"] == "Same-sex"))
        ]
        written = plot_degree_heatmap_evolution(df, windows=WINDOWS_2, output_dir=str(tmp_path))
        assert os.path.exists(written[0])


# ── Integration with the full pipeline ───────────────────────────────────────


class TestIntegrationWithRealMetrics:
    def test_runs_against_real_metrics_output(self, tmp_path):
        """End-to-end: simulate, compute degree_by_demographic, plot."""
        from partnersim_dynet.config import PartnershipConfig
        from partnersim_dynet.generator import PartnershipGenerator
        from partnersim_dynet.network import (
            ActiveIntervals,
            degree_by_demographic_over_time,
            prepare_partnerships,
        )

        total_timesteps = 200
        cfg = PartnershipConfig(num_agents=200, total_timesteps=total_timesteps)
        gen = PartnershipGenerator(cfg, seed=42)
        df = gen.simulate_partnerships()
        log = gen.get_agent_log()
        active = ActiveIntervals.from_agent_log(log, total_timesteps=total_timesteps)
        arr = prepare_partnerships(df, total_timesteps=total_timesteps)

        demo = degree_by_demographic_over_time(arr, active, log, total_timesteps=total_timesteps)

        # 200-step sim: discard first/last 10 as a proportional stand-in
        # for the 50-step rule, then split remainder into 4 equal windows
        analysis_start, analysis_end = 11, 190
        block_width = (analysis_end - analysis_start + 1) // 4  # 45
        windows = [
            (analysis_start + i * block_width, analysis_start + (i + 1) * block_width - 1)
            for i in range(4)
        ]

        written = plot_degree_heatmap_evolution(
            demo,
            windows=windows,
            output_dir=str(tmp_path),
            formats=OutputFormats.all_enabled(),
        )
        assert len(written) == 3
        for p in written:
            assert os.path.exists(p)
            assert os.path.getsize(p) > 1000
