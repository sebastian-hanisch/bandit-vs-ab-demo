"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Scrollen auf dem Handy nicht zoomt."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import bva_constants as C

_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), height=360, legend=dict(orientation="h", y=-0.25))


def _finish(fig, title, xtitle=None, ytitle=None, **extra):
    fig.update_layout(title=dict(text=title, x=0), xaxis=dict(title=xtitle, fixedrange=True),
                      yaxis=dict(title=ytitle, fixedrange=True), **{**_LAYOUT, **extra})
    return fig


def round_shares(arms, m, k=3):
    """Anteil der Fenster je Regel in jeder Runde: Form (Runden, K); x = Mitte der Runde in Fenstern."""
    t_ = len(arms)
    edges = list(range(0, t_, m)) + [t_]
    shares = np.array([[np.mean(arms[a:b] == j) for j in range(k)] for a, b in zip(edges[:-1], edges[1:])])
    mid = np.array([(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])])
    return mid, shares


def build_allocation(arms_ts, arms_abn, m):
    """Wohin gehen die Fenster? Links Bandit, rechts A/B/n: Anteil je Regel in jeder Runde."""
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Bandit (Thompson-Sampling)", "A/B/n: 30 % Test, dann die beste"))
    for col, arms in ((1, arms_ts), (2, arms_abn)):
        mid, sh = round_shares(arms, m)
        for j in range(sh.shape[1]):
            fig.add_trace(go.Scatter(x=mid, y=100 * sh[:, j], mode="lines", stackgroup=f"s{col}", name=C.ARM_LABELS[j],
                                     legendgroup=C.ARM_LABELS[j], showlegend=(col == 1), line=dict(width=0.5, color=C.ARM_COLORS[j]),
                                     fillcolor=C.ARM_COLORS[j]), row=1, col=col)
    fig.update_layout(title=dict(text="Wohin gehen die Standort-Fenster? (Anteil je Runde in %)", x=0), height=360,
                      margin=dict(l=10, r=10, t=70, b=10), legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(title_text="Fenster", fixedrange=True)
    fig.update_yaxes(fixedrange=True, range=[0, 100])
    return fig


def build_regret(curves):
    """Kumulierter Verlust gegenüber der besten Regel. curves: {Strategie: Array (T,) in Prozentpunkten x Fenster}."""
    fig = go.Figure()
    for name, y in curves.items():
        fig.add_trace(go.Scatter(x=np.arange(1, len(y) + 1), y=y, mode="lines", name=C.STRATEGY_SHORT[name],
                                 line=dict(color=C.STRATEGY_COLORS[name], width=2.5)))
    return _finish(fig, "Verlust gegenüber „immer die beste Regel“ (kumuliert)", "Fenster", "Prozentpunkte Servicegrad × Fenster")


def build_estimates(rows, truth):
    """Schätzung von C minus A mit 95-%-Intervall je Auswertung (ein Experiment). rows: [(Beschriftung, Farbe, diff, lo, hi)]."""
    fig = go.Figure()
    labels = [r[0] for r in rows][::-1]
    for i, (label, color, diff, lo, hi) in enumerate(reversed(rows)):
        fig.add_trace(go.Scatter(x=[100 * diff], y=[i], mode="markers", showlegend=False,
                                 error_x=dict(type="data", symmetric=False, array=[100 * (hi - diff)], arrayminus=[100 * (diff - lo)],
                                              thickness=3, width=9, color=color),
                                 marker=dict(color=color, size=11)))
    fig.add_trace(go.Scatter(x=[100 * truth, 100 * truth], y=[-0.6, len(rows) - 0.4], mode="lines", name="wahre Wirkung",
                             line=dict(color=C.COLOR_TRUTH, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[0, 0], y=[-0.6, len(rows) - 0.4], mode="lines", showlegend=False, line=dict(color="gray", width=1.2)))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(rows))), ticktext=labels, range=[-0.6, len(rows) - 0.4])
    return _finish(fig, "C gegen A: Schätzung und 95-%-Intervall (gestrichelt: wahre Wirkung)", "Prozentpunkte Servicegrad", None,
                   height=380, margin=dict(l=10, r=10, t=40, b=10))


def build_loss(rows, best_mean):
    """Was kostet die Strategie? Mittlerer Servicegrad der Experiment-Fenster gegenüber der besten Regel (Prozentpunkte)."""
    names = list(rows)
    fig = go.Figure(go.Bar(x=[C.STRATEGY_SHORT[n] for n in names], y=[100 * (best_mean - rows[n]["reached"]) for n in names],
                           marker_color=[C.STRATEGY_COLORS[n] for n in names],
                           text=[f"{100 * (best_mean - rows[n]['reached']):.2f}".replace(".", ",") for n in names], textposition="outside"))
    fig.update_yaxes(rangemode="tozero")
    return _finish(fig, "Verlust gegenüber „immer die beste Regel“ (Prozentpunkte Servicegrad, Mittel über das Experiment)", None,
                   "Prozentpunkte")


def build_rates(items, aa):
    """Befundraten: items [(Beschriftung, Rate, Farbe)]; aa True = Fehlalarm, sonst Teststärke."""
    fig = go.Figure(go.Bar(x=[i[0] for i in items], y=[100 * i[1] for i in items], marker_color=[i[2] for i in items],
                           text=[f"{100 * i[1]:.1f}".replace(".", ",") + " %" for i in items], textposition="outside"))
    if aa:
        fig.add_hline(y=5, line=dict(color="gray", dash="dot"), annotation_text="Soll 5 %", annotation_position="top right")
    fig.update_yaxes(range=[0, 110])
    return _finish(fig, "Fehlalarm ohne Wirkung (Anteil der Experimente)" if aa else "Teststärke: Anteil der Experimente mit Befund",
                   None, "Anteil in %", height=380)


def build_estimate_spread(items, truth):
    """Schätzung von C minus A über viele Experimente: Mittel ± eine Standardabweichung. items [(Beschriftung, mean, sd, Farbe)]."""
    fig = go.Figure()
    for lab, mean, sd, color in items:
        fig.add_trace(go.Scatter(x=[lab], y=[100 * mean], mode="markers", showlegend=False, marker=dict(color=color, size=12),
                                 error_y=dict(type="data", array=[100 * sd], thickness=3, width=9, color=color)))
    fig.add_hline(y=100 * truth, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="wahre Wirkung",
                  annotation_position="top left")
    return _finish(fig, "Geschätzte Wirkung C gegen A: Mittel ± Standardabweichung über die Experimente", None, "Prozentpunkte")


def build_trend(trend_pp, series, title):
    """Fehlalarmrate gegen den Trend. series: {Beschriftung: (Raten, Farbe, Strich)}."""
    fig = go.Figure()
    for label, (rates, color, dash) in series.items():
        fig.add_trace(go.Scatter(x=trend_pp, y=[100 * r for r in rates], mode="lines+markers", name=label,
                                 line=dict(color=color, width=3, dash=dash)))
    fig.add_hline(y=5, line=dict(color="gray", dash="dot"), annotation_text="Soll 5 %", annotation_position="bottom right")
    fig.update_yaxes(range=[0, 65])
    return _finish(fig, title, "Trend der Gesamtlage über das Experiment (Prozentpunkte Servicegrad)", "Fehlalarmrate in %")


def build_horizon(ts, series, title):
    fig = go.Figure()
    for label, (rates, color, dash) in series.items():
        fig.add_trace(go.Scatter(x=ts, y=[100 * r for r in rates], mode="lines+markers", name=label,
                                 line=dict(color=color, width=3, dash=dash)))
    fig.update_yaxes(range=[0, 105])
    return _finish(fig, title, "Standort-Fenster im Experiment", "Teststärke in %")
