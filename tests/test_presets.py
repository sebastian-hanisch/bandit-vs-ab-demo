"""Presets: Vollständigkeit, gültige Werte, Einrasten der Permalink-Werte."""

import bva_constants as C
import bva_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_on_the_grid():
    for preset in C.PRESETS.values():
        assert C.ZB_MIN <= preset["z"] <= C.ZB_MAX
        steps = (preset["z"] - C.ZB_MIN) / C.ZB_STEP
        assert abs(steps - round(steps)) < 1e-9
        assert C.T_MIN <= preset["T"] <= C.T_MAX and (preset["T"] - C.T_MIN) % C.T_STEP == 0
        assert C.M_MIN <= preset["m"] <= C.M_MAX and (preset["m"] - C.M_MIN) % C.M_STEP == 0
        assert C.TREND_PP_MIN <= preset["trend"] <= C.TREND_PP_MAX
        assert 0 <= preset["seed"] <= C.SEED_MAX


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_labels_are_complete():
    assert set(C.STRATEGY_LABELS) == set(C.STRATEGY_SHORT) == set(C.STRATEGY_COLORS)
    assert len(C.ARM_LABELS) == len(C.ARM_COLORS) == 3


def test_best_rule_factors_are_on_the_pool_grid():
    from bva_bandit import arm_z
    for z in (1.0, 1.2, 1.4, 1.6, 1.8):
        assert all(a in C.Z_GRID for a in arm_z(z))


def test_permalink_values_are_clamped_and_snapped_to_the_slider_grid():
    t_spec = P.SETTING_SPECS["t_slider"]
    assert P.clamp_to_spec(t_spec, 640) == 600 and P.clamp_to_spec(t_spec, 1) == 150 and P.clamp_to_spec(t_spec, 5000) == 900
    z_spec = P.SETTING_SPECS["z_slider"]
    assert P.clamp_to_spec(z_spec, 1.31) == 1.4 and P.clamp_to_spec(z_spec, 9.0) == 1.8
    assert P.clamp_to_spec(P.SETTING_SPECS["trend_slider"], 17) == 4
