"""Bandit statt A/B-Test - was kostet das Lernen beim Zuteilen? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der Konzepte-Linie "Experimente und Wirkungsmessung": Ein Bandit (Thompson-Sampling) verteilt die Standort-
Fenster nach und nach auf die Regel, die besser aussieht, und verliert dadurch weniger als ein gleichverteilter Test. Die
Frage dieses Stücks ist die Auswertung: Was gilt der Test noch, wenn die Zuteilung von den Ergebnissen abhängt? Siehe README.

Lauffähig mit: streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st

import bva_constants as C
from bva_evaluation import load_pools, nearest_z, regret_curves, run_one, single_estimates, summarize, true_means
from bva_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed,
                         sync_query_params)
from bva_visualization import (build_allocation, build_estimate_spread, build_estimates, build_horizon, build_loss, build_rates,
                               build_regret, build_trend)

st.set_page_config(page_title="Bandit statt A/B-Test – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _pools():
    return load_pools()


@st.cache_data(show_spinner=False)
def _sweeps():
    data = json.loads((Path(__file__).resolve().parent / "precomputed_sweeps.json").read_text(encoding="utf-8"))
    return data["cells"]


@st.cache_data(show_spinner=False)
def _one(z, t, m, trend, seed):
    one = run_one(_pools(), {"z": z, "T": t, "m": m, "trend": trend, "seed": seed})
    return {"arms": {k: r["arms"][0] for k, r in one["runs"].items()}, "regret": regret_curves(one),
            "estimates": single_estimates(one), "means": one["means"], "n_test": one["runs"]["abn"]["n_test"]}


@st.cache_data(show_spinner=False)
def _many(z, t, m, trend):
    return summarize(_pools(), z, t, m, trend, reps=C.LIVE_REPS)


def _cell(sweep, z, t=None, m=None, trend=None):
    for c in _sweeps():
        if c["sweep"] == sweep and abs(c["z"] - z) < 1e-9 and (t is None or c["T"] == t) and (m is None or c["m"] == m) \
                and (trend is None or abs(c["trend"] - trend) < 1e-9):
            return c
    raise KeyError((sweep, z, t, m, trend))


st.title("🎰 Bandit statt A/B-Test: was kostet das Lernen beim Zuteilen?")
st.markdown(
    """
Ein A/B-Test teilt **gleichmäßig** zu und lernt dabei absichtlich auch auf der schlechteren Regel. Ein **Bandit** verschiebt
die Fenster nach und nach dorthin, wo es besser aussieht, und verliert so weniger. Das ist der Vorteil, den jede Einführung
verspricht. Die Frage dieses Stücks ist der Preis: **Sobald die Zuteilung von den bisherigen Ergebnissen abhängt, ist der
naive Test nicht mehr der, den man kennt.** Hier laufen vier Strategien auf denselben möglichen Ergebnissen gegeneinander,
mit bekannter Wahrheit.
"""
)
st.caption(
    "Neuntes Stück der Linie „Experimente und Wirkungsmessung“, aufbauend auf "
    "[ab-test-demo](https://sebastianhanisch-ab-test-demo.streamlit.app/) (gleiches Vehikel: Standort-Fenster mit Servicegrad) "
    "und [bandit-demo](https://sebastianhanisch-bandit-demo.streamlit.app/) (Regret, Strategien). Dort geht es um den Verlust; "
    "hier um die **Inferenz** nach adaptiver Zuteilung."
)

with st.expander("So ist das Experiment aufgebaut", expanded=True):
    st.markdown(
        """
- **Drei Regeln:** A ist die bestehende (Sicherheitsfaktor 1,0), C die neue mit dem eingestellten Faktor, B liegt in der Mitte.
  C ist die beste Regel; ob man das weiß, entscheidet das Experiment.
- **Einheit:** ein Standort-Fenster (vier Wochen) mit Servicegrad als Ergebnis. Die Fenster treffen in **Runden** zu je *m*
  Fenstern ein; die Zuteilung einer Runde sieht nur die Ergebnisse früherer Runden.
- **Vier Strategien:** *gleichverteilt* (jedes Fenster zufällig auf eine Regel), *A/B/n* (30 % Testphase gleichverteilt, danach
  alles auf die Regel mit dem besten Mittel), *Bandit* (Thompson-Sampling) und *Bandit mit Boden* (10 % der Fenster bleiben
  gleichverteilt).
- **Gleiche Zufallszahlen:** jedes Fenster hat für jede Regel ein festes mögliches Ergebnis; die Strategie bestimmt nur, welches
  sie sieht. So sind Unterschiede zwischen den Strategien keine Würfelunterschiede.
- **Die Frage an die Daten:** Ist C besser als A, und um wie viel? Beantwortet mit dem Welch-Test auf allen gesammelten Fenstern
  (naiv), auf den gleichverteilten Fenstern allein, und mit einer Gewichtung durch die Zuteilungswahrscheinlichkeiten.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{i}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])
st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Wahrheit**")
    z = st.slider("Sicherheitsfaktor der besten Regel C", *bounds("z_slider"), step=C.ZB_STEP, key="z_slider", format="%.1f",
                  help="A fährt 1,0, B die Mitte. Bei 1,0 sind alle drei Regeln gleich (A/A/A): jeder Befund ist ein Fehlalarm.")
    trend_pp = st.slider("Trend der Gesamtlage (Prozentpunkte)", *bounds("trend_slider"), step=1, key="trend_slider",
                         help="Alle Regeln verschieben sich gemeinsam über die Dauer des Experiments (etwa durch die Saison): "
                              "von minus bis plus die Hälfte dieses Werts.")
    st.markdown("**Experiment**")
    t = st.slider("Standort-Fenster insgesamt", *bounds("t_slider"), step=C.T_STEP, key="t_slider",
                  help="Wie viele Fenster das Experiment insgesamt zuteilt.")
    m = st.slider("Fenster je Runde", *bounds("m_slider"), step=C.M_STEP, key="m_slider",
                  help="Die Ergebnisse einer Runde sind erst für die nächste sichtbar (ein Fenster dauert vier Wochen).")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1,
                           key="seed_input", help="Bestimmt die Fenster des einen Experiments unten.")
    st.button("🎲 Neues Experiment würfeln", on_click=randomize_seed)

z, t, m, trend_pp, seed = nearest_z(float(z)), int(t), int(m), int(trend_pp), int(seed)
trend = trend_pp / 100.0
sync_query_params({"z_slider": z, "t_slider": t, "m_slider": m, "trend_slider": trend_pp, "seed_input": seed})
aa = z <= C.Z_A
pools = _pools()
means = true_means(pools, z)
truth = float(means[2] - means[0])
best_mean = float(means.max())

# --- Ein Experiment ------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 🧪 Ein Experiment")
one = _one(z, t, m, trend, seed)
st.caption(
    f"{C.fmt_int(t)} Fenster in Runden zu {m}; wahre Servicegrade A / B / C: "
    f"{C.fmt_pct(means[0])} / {C.fmt_pct(means[1])} / {C.fmt_pct(means[2])}; wahre Wirkung C gegen A: **{C.fmt_pp(truth)}**."
    + (" Alle Regeln sind gleich." if aa else "")
)
st.plotly_chart(build_allocation(one["arms"]["ts"], one["arms"]["abn"], m), width="stretch", key=f"alloc_{z}_{t}_{m}_{trend_pp}_{seed}")
st.plotly_chart(build_regret(one["regret"]), width="stretch", key=f"regret_{z}_{t}_{m}_{trend_pp}_{seed}")

labels = {
    "gleich": ("Gleichverteilt: Welch, alle Fenster", C.STRATEGY_COLORS["gleich"]),
    "abn": ("A/B/n: Welch, nur Testphase", C.STRATEGY_COLORS["abn"]),
    "ts": ("Bandit: Welch, alle Fenster (naiv)", C.STRATEGY_COLORS["ts"]),
    "ts_boden": ("Bandit + Boden: Welch, alle Fenster (naiv)", C.STRATEGY_COLORS["ts_boden"]),
    "ts_boden_clean": ("Bandit + Boden: nur gleichverteilte Fenster", "#17a589"),
    "ts_boden_ipw": ("Bandit + Boden: gewichtet", "#0b5345"),
}
est = one["estimates"]
st.plotly_chart(build_estimates([(labels[e["key"]][0], labels[e["key"]][1], e["diff"], e["lo"], e["hi"]) for e in est], truth),
                width="stretch", key=f"est_{z}_{t}_{m}_{trend_pp}_{seed}")
rows_md = ["| Auswertung | Schätzung C − A | 95-%-Intervall | p-Wert |", "|---|---|---|---|"]
for e in est:
    rows_md.append(f"| {labels[e['key']][0]} | {C.fmt_pkt(e['diff'], 2)} | {C.fmt_pkt(e['lo'], 2)} bis {C.fmt_pkt(e['hi'], 2)} | "
                   f"{C.fmt_num(e['p'], 3)} |")
st.markdown("\n".join(rows_md))
st.caption("Ein einzelnes Experiment kann zufällig typisch oder untypisch sein; die tausend Wiederholungen darunter zeigen, wie "
           "oft das gutgeht.")

# --- Viele Experimente ---------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📐 Tausend Experimente: Verlust und Auswertung")
with st.spinner("Rechne …"):
    many = _many(z, t, m, trend)
rows = many["rows"]
loss = {k: best_mean - rows[k]["reached"] for k in rows}
st.plotly_chart(build_loss(rows, best_mean), width="stretch", key=f"loss_{z}_{t}_{m}_{trend_pp}")

items = [
    ("Gleichverteilt", rows["gleich"]["reject"], C.STRATEGY_COLORS["gleich"]),
    ("A/B/n (Testphase)", rows["abn"]["reject"], C.STRATEGY_COLORS["abn"]),
    ("Bandit (naiv)", rows["ts"]["reject"], C.STRATEGY_COLORS["ts"]),
    ("Bandit + Boden (naiv)", rows["ts_boden"]["reject"], C.STRATEGY_COLORS["ts_boden"]),
    ("Bandit + Boden: nur gleichverteilte Fenster", rows["ts_boden"]["clean_reject"], "#17a589"),
    ("Bandit + Boden: gewichtet", rows["ts_boden"]["ipw_reject"], "#0b5345"),
]
st.plotly_chart(build_rates(items, aa), width="stretch", key=f"rates_{z}_{t}_{m}_{trend_pp}")
st.plotly_chart(
    build_estimate_spread(
        [("Gleichverteilt", rows["gleich"]["est"], rows["gleich"]["est_sd"], C.STRATEGY_COLORS["gleich"]),
         ("A/B/n", rows["abn"]["est"], rows["abn"]["est_sd"], C.STRATEGY_COLORS["abn"]),
         ("Bandit (naiv)", rows["ts"]["est"], rows["ts"]["est_sd"], C.STRATEGY_COLORS["ts"]),
         ("Bandit + Boden (naiv)", rows["ts_boden"]["est"], rows["ts_boden"]["est_sd"], C.STRATEGY_COLORS["ts_boden"]),
         ("Bandit + Boden: gewichtet", rows["ts_boden"]["ipw_est"], rows["ts_boden"]["ipw_sd"], "#0b5345")], truth),
    width="stretch", key=f"spread_{z}_{t}_{m}_{trend_pp}")

tab = ["| Strategie | Servicegrad im Experiment | Fenster auf C | " + ("Fehlalarm" if aa else "Teststärke")
       + " | Schätzung C − A (Mittel ± Streuung) | Standardfehler-Treue |", "|---|---|---|---|---|---|"]
for k in ("gleich", "abn", "ts", "ts_boden"):
    r = rows[k]
    tab.append(f"| {C.STRATEGY_SHORT[k]} | {C.fmt_pct(r['reached'], 2)} | {C.fmt_pct(r['share'][2], 0)} | {C.fmt_pct(r['reject'])} | "
               f"{C.fmt_pkt(r['est'], 2)} ± {C.fmt_num(100 * r['est_sd'], 2)} | {C.fmt_num(r['se_fidelity'], 2)} |")
st.markdown("\n".join(tab))
st.caption("Standardfehler-Treue = mittlerer Standardfehler / tatsächliche Streuung der Schätzung (Soll 1; unter 1 heißt: der "
           "Test ist zu zuversichtlich). Der Bandit-Test ohne Boden hat kein gültiges Gegenstück, weil Regeln mit "
           "Zuteilungswahrscheinlichkeit nahe null keine Daten liefern.")

ts, tb, gl, ab = rows["ts"], rows["ts_boden"], rows["gleich"], rows["abn"]
if aa:
    st.error(
        f"**Ohne jeden Unterschied meldet der naive Test nach dem Bandit in {C.fmt_pct(ts['reject'])} der Experimente einen "
        f"Befund** (versprochen 5 %, gleichverteilt {C.fmt_pct(gl['reject'])}); der Standardfehler ist nur zu "
        f"{C.fmt_pct(ts['se_fidelity'], 0)} der tatsächlichen Streuung. Mit Boden und dem Test nur auf den gleichverteilten "
        f"Fenstern sind es {C.fmt_pct(tb['clean_reject'])}, mit der Gewichtung {C.fmt_pct(tb['ipw_reject'])}."
    )
else:
    over = ts["est"] - truth
    st.error(
        f"**Der Bandit verliert im Mittel {C.fmt_num(100 * loss['ts'], 2)} Prozentpunkte Servicegrad gegenüber der besten Regel, "
        f"A/B/n {C.fmt_num(100 * loss['abn'], 2)}, gleichverteilt {C.fmt_num(100 * loss['gleich'], 2)}.** Dafür schätzt der "
        f"naive Test die Wirkung im Mittel um {C.fmt_num(100 * over, 2)} Punkte zu hoch (wahr {C.fmt_pp(truth)}), mit "
        f"{C.fmt_num(ts['est_sd'] / gl['est_sd'], 1)}-facher Streuung der Schätzung; die Teststärke liegt bei "
        f"{C.fmt_pct(ts['reject'], 0)} gegen {C.fmt_pct(gl['reject'], 0)} bei gleichverteilter Zuteilung."
    )

# --- Trend ----------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📈 Wenn die Gesamtlage sich verschiebt")
trend_cells = [_cell("trend", 1.0, trend=tr / 100.0) for tr in range(C.TREND_PP_MIN, C.TREND_PP_MAX + 1)]
series = {
    "Gleichverteilt (naiv)": ([c["rows"]["gleich"]["reject"] for c in trend_cells], C.STRATEGY_COLORS["gleich"], "solid"),
    "Bandit (naiv)": ([c["rows"]["ts"]["reject"] for c in trend_cells], C.STRATEGY_COLORS["ts"], "solid"),
    "Bandit + Boden (naiv)": ([c["rows"]["ts_boden"]["reject"] for c in trend_cells], C.STRATEGY_COLORS["ts_boden"], "solid"),
    "Bandit + Boden: nur gleichverteilte Fenster": ([c["rows"]["ts_boden"]["clean_reject"] for c in trend_cells], "#17a589", "dash"),
    "Bandit + Boden: gewichtet": ([c["rows"]["ts_boden"]["ipw_reject"] for c in trend_cells], "#0b5345", "dot"),
}
st.plotly_chart(build_trend(list(range(C.TREND_PP_MIN, C.TREND_PP_MAX + 1)), series,
                            "Fehlalarm ohne Unterschied (A/A/A, 600 Fenster, 10 je Runde)"),
                width="stretch", key="trend_chart")
c3 = _cell("trend", 1.0, trend=0.03)["rows"]
st.info(
    f"Die Gesamtlage steigt über das Experiment, **alle Regeln gleich**. Gleichverteilt (jedes Fenster unabhängig von der Zeit "
    f"zugeteilt) merkt davon nichts: {C.fmt_pct(c3['gleich']['reject'])} Fehlalarm bei 3 Punkten Trend. Der Bandit teilt dem "
    f"Zufall folgend frühe und späte Fenster verschieden auf die Regeln auf und meldet **{C.fmt_pct(c3['ts']['reject'], 0)}**. "
    f"Der Test auf den gleichverteilten Fenstern ({C.fmt_pct(c3['ts_boden']['clean_reject'])}) und die Gewichtung "
    f"({C.fmt_pct(c3['ts_boden']['ipw_reject'])}) bleiben bei höchstens 5 %: dort hängt die Zuteilung nicht vom Verlauf ab."
)

# --- Kleiner Unterschied braucht Zeit --------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("⏱️ Ein kleiner Unterschied braucht Fenster")
ts_grid = (150, 300, 600, 900)
h_cells = [_cell("horizon", 1.2, t=tt, m=C.DEFAULT_M, trend=0.0) for tt in ts_grid]
series_h = {
    "Gleichverteilt": ([c["rows"]["gleich"]["reject"] for c in h_cells], C.STRATEGY_COLORS["gleich"], "solid"),
    "A/B/n (Testphase)": ([c["rows"]["abn"]["reject"] for c in h_cells], C.STRATEGY_COLORS["abn"], "solid"),
    "Bandit (naiv)": ([c["rows"]["ts"]["reject"] for c in h_cells], C.STRATEGY_COLORS["ts"], "solid"),
    "Bandit + Boden (naiv)": ([c["rows"]["ts_boden"]["reject"] for c in h_cells], C.STRATEGY_COLORS["ts_boden"], "solid"),
    "Bandit + Boden: gewichtet": ([c["rows"]["ts_boden"]["ipw_reject"] for c in h_cells], "#0b5345", "dot"),
}
st.plotly_chart(build_horizon(list(ts_grid), series_h, "Teststärke bei Sicherheitsfaktor 1,2 von C (kleine Wirkung)"),
                width="stretch", key="horizon_chart")
h = h_cells[2]["rows"]
st.info(
    f"Bei 600 Fenstern findet der gleichverteilte Test den kleinen Unterschied in {C.fmt_pct(h['gleich']['reject'], 0)} der "
    f"Experimente, A/B/n in {C.fmt_pct(h['abn']['reject'], 0)} und der Bandit in {C.fmt_pct(h['ts']['reject'], 0)}. Dafür "
    f"erreichen die adaptiven Strategien einen höheren Servicegrad im Experiment ({C.fmt_pct(h['ts']['reached'], 2)} gegen "
    f"{C.fmt_pct(h['gleich']['reached'], 2)}). **Lernen und Verdienen konkurrieren um dieselben Fenster.**"
)

# --- Grenzen --------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Ziel ist das Ergebnis während des Experiments** | Soll das Experiment vor allem entscheiden, welche Regel an 1 000 weitere Standorte geht, zählt die Inferenz, nicht der Verlust; dann ist A/B/n oder Bandit mit Boden die ehrlichere Wahl. | (dieses Stück) |
| **Die Gesamtlage ist stabil** | Ein gemeinsamer Trend macht den naiven Test nach adaptiver Zuteilung zu einem Zufallsgenerator. | (dieses Stück) |
| **Die Regeln beeinflussen sich nicht** | Teilen Standorte Kundschaft oder Lager, misst der Vergleich etwas anderes. | Interferenz und Switchback |
| **Ergebnisse liegen rechtzeitig vor** | Ein Fenster dauert vier Wochen; je größer die Runde, desto weniger kann der Bandit lernen (Regler „Fenster je Runde“). | (dieses Stück) |
| **Ein Ergebnis, ein Test** | Mehrere Kennzahlen oder mehrere neue Regeln vervielfachen die Tests. | Viele Kennzahlen: Mehrfachtests und Gewinnerfluch |
| **Die Zuteilung hängt nur vom Verlauf ab** | Hängt sie von Merkmalen des Standorts ab (kontextueller Bandit), braucht die Auswertung Logdaten und Wahrscheinlichkeiten. | Auswertung aus Logdaten (Off-Policy) |
"""
)
st.caption(
    "Die Zahlen stammen aus dem erfundenen Vehikel dieses Stücks (Standort-Fenster mit Servicegrad Typ 2, drei Regeln); "
    "Trend und Runden sind Modellannahmen, keine Messung am Betrieb."
)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Thompson-Sampling (Normalmodell).** Je Regel $k$ aus den bisherigen $n_k$ Fenstern das Mittel $\bar y_k$ und die Streuung
$s_k$; für jedes neue Fenster eine Ziehung $\theta_k \sim \mathcal N(\bar y_k,\, s_k^2/n_k)$ und die Regel mit der größten
Ziehung. Anlauf: bis jede Regel fünf Fenster hat, gleichverteilt. **Boden:** mit Wahrscheinlichkeit $q = 0{,}1$ ein
gleichverteiltes Fenster, sonst Thompson.

**Warum der naive Test kippt.** Welch-Test und Konfidenzintervall setzen voraus, dass die Zuteilung unabhängig von den
Ergebnissen ist. Beim Bandit hängt sie vom Verlauf ab: Regeln, die zufällig früh gut aussehen, bekommen mehr Fenster; Regeln mit
zufällig schlechtem Start bekommen kaum noch welche und werden nicht mehr korrigiert. Das verzerrt die Mittel und macht den
Standardfehler zu klein.

**Gewichtung.** Mit der protokollierten Wahrscheinlichkeit $\pi_{t}$, mit der die Regel des Fensters $t$ gewählt wurde, ist
$\hat\mu_k = \sum_t \mathbb 1[A_t = k]\, y_t/\pi_t \,/\, \sum_t \mathbb 1[A_t = k]/\pi_t$ (normiert, Hájek). Der Standardfehler
folgt aus der Linearisierung $\sum_t \mathbb 1[A_t=k]\,(y_t-\hat\mu_k)^2/\pi_t^2 \,/\, (\sum_t \mathbb 1[A_t=k]/\pi_t)^2$. Das
braucht $\pi_t > 0$ für jede Regel: ohne Boden kann $\pi_t$ beliebig klein werden, mit Boden ist es mindestens $q/3$. Die
Gewichte werden bei $10^{-3}$ gekappt.

**Test auf den gleichverteilten Fenstern.** Fenster aus dem Anlauf und dem Boden sind unabhängig von allen Ergebnissen
zugeteilt; auf ihnen gilt der gewöhnliche Welch-Test. Preis: nur diese Fenster tragen Information.

Implementiert in `bva_bandit.py` (Strategien), `bva_evaluation.py` (Tests, Gewichtung, Zusammenfassungen), `bva_vehicle.py`
(Standort-Fenster), `generate_precomputed.py` (Pools und Sweeps).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
