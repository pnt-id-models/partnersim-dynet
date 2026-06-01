"""Tests for the plot style/save infrastructure."""

from __future__ import annotations

import os

import matplotlib

# Use a non-interactive backend so tests don't try to open windows
matplotlib.use("Agg")


import matplotlib.pyplot as plt
import pytest

from partnersim_dynet.network.plots import (
    PALETTE,
    NetworkPalette,
    OutputFormats,
    publication_style,
    save_figure,
)

# Palette


class TestPalette:
    def test_module_level_palette_exists(self):
        assert isinstance(PALETTE, NetworkPalette)

    def test_palette_is_frozen(self):
        """Mutating PALETTE should raise (dataclass frozen=True)."""
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):  # FrozenInstanceError
            PALETTE.avg_degree = "#000000"  # type: ignore[misc]

    def test_orientation_color_known(self):
        assert PALETTE.orientation_color("Opposite-sex") == PALETTE.orientation_opposite
        assert PALETTE.orientation_color("Same-sex") == PALETTE.orientation_same
        assert PALETTE.orientation_color("Bisexual") == PALETTE.orientation_bisexual

    def test_orientation_color_unknown_returns_fallback(self):
        assert PALETTE.orientation_color("Unknown") == "#999999"

    def test_sex_shape_known(self):
        assert PALETTE.sex_shape("Male") == "o"
        assert PALETTE.sex_shape("Female") == "s"

    def test_metric_colors_distinct(self):
        """The four timeseries metrics should all have different colors."""
        colors = [
            PALETTE.avg_degree,
            PALETTE.max_degree,
            PALETTE.transitivity,
            PALETTE.avg_path_length,
        ]
        assert len(set(colors)) == 4


# publication_style context manager


class TestPublicationStyle:
    def test_rcparams_changed_inside_context(self):
        """Inside the context, our settings should be active."""
        with publication_style():
            assert plt.rcParams["savefig.dpi"] == 300
            assert plt.rcParams["axes.spines.top"] is False

    def test_rcparams_restored_after_context(self):
        """The whole point: settings outside should NOT be mutated."""
        original_dpi = plt.rcParams["savefig.dpi"]
        with publication_style():
            pass
        # After exit, dpi is whatever it was before
        assert plt.rcParams["savefig.dpi"] == original_dpi

    def test_serif_font_by_default(self):
        with publication_style():
            assert plt.rcParams["font.family"] == ["serif"]

    def test_sans_serif_font_when_helvetica(self):
        with publication_style(font="Helvetica"):
            assert plt.rcParams["font.family"] == ["sans-serif"]


# OutputFormats


class TestOutputFormats:
    def test_default_is_png_only(self):
        f = OutputFormats()
        assert f.png is True
        assert f.pdf is False
        assert f.svg is False

    def test_all_enabled(self):
        f = OutputFormats.all_enabled()
        assert f.png and f.pdf and f.svg

    def test_any_enabled_true_when_any_set(self):
        assert OutputFormats(png=True).any_enabled()
        assert OutputFormats(pdf=True, png=False).any_enabled()
        assert OutputFormats(svg=True, png=False).any_enabled()

    def test_any_enabled_false_when_all_off(self):
        assert OutputFormats(png=False, pdf=False, svg=False).any_enabled() is False


# save_figure


class TestSaveFigure:
    def _make_fig(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        return fig

    def test_default_writes_png_only(self, tmp_path):
        fig = self._make_fig()
        base = str(tmp_path / "test_plot")
        written = save_figure(fig, base)
        plt.close(fig)

        assert written == [f"{base}.png"]
        assert os.path.exists(f"{base}.png")
        assert not os.path.exists(f"{base}.pdf")
        assert not os.path.exists(f"{base}.svg")

    def test_all_three_formats(self, tmp_path):
        fig = self._make_fig()
        base = str(tmp_path / "test_plot")
        written = save_figure(fig, base, OutputFormats.all_enabled())
        plt.close(fig)

        assert len(written) == 3
        for ext in ("png", "pdf", "svg"):
            assert os.path.exists(f"{base}.{ext}")

    def test_only_pdf(self, tmp_path):
        fig = self._make_fig()
        base = str(tmp_path / "test_plot")
        save_figure(fig, base, OutputFormats(png=False, pdf=True))
        plt.close(fig)

        assert not os.path.exists(f"{base}.png")
        assert os.path.exists(f"{base}.pdf")

    def test_creates_parent_directory(self, tmp_path):
        fig = self._make_fig()
        base = str(tmp_path / "subdir1" / "subdir2" / "test_plot")
        save_figure(fig, base)
        plt.close(fig)

        assert os.path.exists(f"{base}.png")

    def test_all_formats_disabled_raises(self, tmp_path):
        fig = self._make_fig()
        base = str(tmp_path / "test_plot")
        with pytest.raises(ValueError, match="all formats disabled"):
            save_figure(fig, base, OutputFormats(png=False, pdf=False, svg=False))
        plt.close(fig)

    def test_files_have_nonzero_size(self, tmp_path):
        """A real check: the saved file should actually have content."""
        fig = self._make_fig()
        base = str(tmp_path / "test_plot")
        written = save_figure(fig, base)
        plt.close(fig)

        for p in written:
            assert os.path.getsize(p) > 100, f"{p} suspiciously small"
