import numpy as np
import polars as pl
import pytest

from src_polars.constants import C_LIGHT_MPS


def test_differential_code_corrections():
    from src_polars.gnss import apply_differential_corrections

    df1 = pl.DataFrame(
        {
            "epoch": [0, 0, 1, 1],
            "constellation": ["G"] * 4,
            "prn": [1, 2, 1, 2],
            "rnx_obs_identifier": ["1C"] * 4,
            "C_obs_m": [3, 4, 5, 6],
            "L_obs_cycles": [
                cycles * 1575.42e6 / C_LIGHT_MPS for cycles in [3, 4, 5, 6]
            ],
            "carrier_frequency_hz": [1575.42e6] * 4,
            "sat_pos_x_m": [0] * 4,
            "sat_pos_y_m": [0] * 4,
            "sat_pos_z_m": [0] * 4,
        }
    )
    df2 = df1.clone()
    pos_base = np.array([0] * 3)

    df_result = apply_differential_corrections(df1, df2, pos_base)
    for col in df1.columns + ["C_obs_corr_m"]:
        assert col in df_result.columns, f"{col} not in the resulting dataframe "
    assert df_result.get_column("C_obs_corr_m").to_numpy() == pytest.approx(
        np.array([0] * 4)
    )

    # remove one row from df2
    df_result2 = apply_differential_corrections(df1, df2[1:], pos_base)
    assert not (df_result2.get_column("C_obs_corr_m").is_nan()).any(), (
        "missing row should not result in NaN"
    )
    assert len(df_result2) == 3, (
        "Only the rows common to both dataframes should be present"
    )
    assert df_result2.get_column("C_obs_corr_m").to_numpy() == pytest.approx(
        np.array([0] * 3)
    )


def test_corrected_code_model():
    """
    Test different df_prx with corrections present.
    They should not be taken into account in the corrected code observation model.
    """
    from src_polars.gnss import obs_model_corrected_code

    # test geometric distance
    df = pl.DataFrame(
        {
            "sat_pos_x_m": [1, 0, 0],
            "sat_pos_y_m": [0, 1, 0],
            "sat_pos_z_m": [0, 0, 1],
            "sat_clock_offset_m": [0] * 3,
            "sagnac_effect_m": [0] * 3,
            "relativistic_clock_effect_m": [0] * 3,
            "sat_code_bias_m": [0] * 3,
            "iono_delay_m": [0] * 3,
            "tropo_delay_m": [0] * 3,
        }
    )
    rx_pos = [0, 0, 0]
    rx_clk = 0
    assert (
        obs_model_corrected_code(df, rx_pos, rx_clk) == np.array([1.0] * 3)
    ).all(), "geometric distance wrongly computed"

    # test rx clock bias
    rx_clk = 1
    assert (
        obs_model_corrected_code(df, rx_pos, rx_clk) == np.array([2.0] * 3)
    ).all(), "issue with rx_clk"
    assert (obs_model_corrected_code(df, rx_pos) == np.array([1.0] * 3)).all(), (
        "rx_clk default value should be 0"
    )

    # test corrections with plus sign
    df = pl.DataFrame(
        {
            "sat_pos_x_m": [1, 0, 0],
            "sat_pos_y_m": [0, 1, 0],
            "sat_pos_z_m": [0, 0, 1],
            "sat_clock_offset_m": [0] * 3,
            "sagnac_effect_m": [1] * 3,
            "relativistic_clock_effect_m": [0] * 3,
            "sat_code_bias_m": [1] * 3,
            "iono_delay_m": [1] * 3,
            "tropo_delay_m": [1] * 3,
        }
    )
    rx_pos = [0, 0, 0]
    rx_clk = 0
    assert (obs_model_corrected_code(df, rx_pos, rx_clk) == np.array([1.0] * 3)).all()

    # test corrections with minus sign
    df = pl.DataFrame(
        {
            "sat_pos_x_m": [1, 0, 0],
            "sat_pos_y_m": [0, 1, 0],
            "sat_pos_z_m": [0, 0, 1],
            "sat_clock_offset_m": [1] * 3,
            "sagnac_effect_m": [0] * 3,
            "relativistic_clock_effect_m": [1] * 3,
            "sat_code_bias_m": [0] * 3,
            "iono_delay_m": [0] * 3,
            "tropo_delay_m": [0] * 3,
        }
    )
    rx_pos = [0, 0, 0]
    rx_clk = 0
    assert (obs_model_corrected_code(df, rx_pos, rx_clk) == np.array([1.0] * 3)).all()
