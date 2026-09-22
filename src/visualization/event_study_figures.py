"""Event-time figures for the realised market response (Revision 2, T11)."""

from __future__ import annotations

from .result_figures import (H_MED, PALETTE, WIDTH_FULL, save_figure, stamp)

LABELS = {"return": "cumulative average abnormal return (log points)",
          "volume": "cumulative average abnormal log volume"}


def plot_event_time_caar(table, label: str, name: str = None):
    """One series: the cumulative average abnormal value with its confidence band."""
    import matplotlib.pyplot as plt

    frame = table[table["series"] == label].sort_values("event_time")
    fig, ax = plt.subplots(figsize=(WIDTH_FULL, H_MED))
    ax.axhline(0.0, color=PALETTE["ink"], lw=0.8)
    ax.fill_between(frame.event_time, frame.ci_low, frame.ci_high,
                    color=PALETTE["blue"], alpha=0.18, label="95% cross-sectional interval")
    ax.plot(frame.event_time, frame.caar, color=PALETTE["blue"], marker="o", ms=3,
            label="cumulative average abnormal value")
    ax.set_xlabel("event time, trading sessions from the prediction origin (k = 1 is the first)")
    ax.set_ylabel(LABELS.get(label, label))
    ax.set_title(f"Realised {label} response in event time")
    ax.legend(fontsize=6.5, loc="best")

    at5 = frame[frame.event_time == 5]
    if len(at5):
        row = at5.iloc[0]
        stamp(fig, f"{int(row.n_events)} events | CAAR(1,5) {row.caar:+.4f} | "
                   f"BMP t {row.t_bmp:+.2f} p {row.p_bmp:.3f} | "
                   f"Corrado t {row.t_corrado:+.2f} p {row.p_corrado:.3f}")
    save_figure(fig, name or f"fig_32_event_time_{label}")
    return fig


def plot_event_time_panel(table, name: str = "fig_33_event_time_tests"):
    """Both series beside the test statistics, so significance is read off the same page."""
    import matplotlib.pyplot as plt

    series = [s for s in ("return", "volume") if (table["series"] == s).any()]
    fig, axes = plt.subplots(2, len(series), figsize=(WIDTH_FULL, H_MED * 1.5),
                             sharex=True, squeeze=False)
    for column, label in enumerate(series):
        frame = table[table["series"] == label].sort_values("event_time")
        top = axes[0][column]
        top.axhline(0.0, color=PALETTE["ink"], lw=0.8)
        top.fill_between(frame.event_time, frame.ci_low, frame.ci_high,
                         color=PALETTE["blue"], alpha=0.18)
        top.plot(frame.event_time, frame.caar, color=PALETTE["blue"], marker="o", ms=2.5)
        top.set_title(label, fontsize=8)
        top.set_ylabel(LABELS.get(label, label), fontsize=6.5)

        bottom = axes[1][column]
        for statistic, colour, marker in (("t_bmp", "orange", "o"),
                                          ("t_corrado", "green", "s"),
                                          ("t_kolari_pynnonen", "vermil", "^")):
            bottom.plot(frame.event_time, frame[statistic], color=PALETTE[colour],
                        marker=marker, ms=2.5, lw=1.0,
                        label=statistic.replace("t_", "").replace("_", " "))
        for level in (-1.96, 1.96):
            bottom.axhline(level, color=PALETTE["grey"], lw=0.8, ls="--")
        bottom.axhline(0.0, color=PALETTE["ink"], lw=0.8)
        bottom.set_xlabel("event time (sessions)")
        bottom.set_ylabel("test statistic", fontsize=6.5)
        if column == 0:
            bottom.legend(fontsize=6, loc="best")

    fig.suptitle("Realised response and its test statistics, dashed lines at the 5% level",
                 fontsize=9)
    stamp(fig, "a statistic inside the dashed lines is not distinguishable from no reaction")
    save_figure(fig, name)
    return fig
