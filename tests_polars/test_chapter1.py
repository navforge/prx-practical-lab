import polars as pl
import numpy as np
import pytest

from src_polars.helpers import repo_root


def test_uncorrected_code_model():
    from src_polars.gnss import obs_model_code

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
    computed = obs_model_code(df, rx_pos, rx_clk)
    assert isinstance(computed, np.ndarray), (
        "output of obs_model_code is not a numpy array"
    )
    assert computed.shape == (3,), "output of obs_model_code has wrong shape"
    assert (computed == np.array([1.0] * 3)).all(), (
        "geometric distance wrongly computed"
    )

    # test rx clock bias
    rx_clk = 1
    assert (obs_model_code(df, rx_pos, rx_clk) == np.array([2.0] * 3)).all(), (
        "issue with rx_clk"
    )
    assert (obs_model_code(df, rx_pos) == np.array([1.0] * 3)).all(), (
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
    assert (obs_model_code(df, rx_pos, rx_clk) == np.array([5.0] * 3)).all(), (
        "issue among sagnac_effect_m, sat_code_bias_m, iono_delay_m, tropo_delay_m"
    )

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
    assert (obs_model_code(df, rx_pos, rx_clk) == np.array([-1.0] * 3)).all(), (
        "issue among sat_clock_offset_m, relativistic_clock_effect_m"
    )


def test_jacobian_code():
    from src_polars.gnss import jacobian_code

    df = pl.DataFrame(
        {
            "sat_pos_x_m": [1, 0, 0],
            "sat_pos_y_m": [0, 1, 0],
            "sat_pos_z_m": [0, 0, 1],
        }
    )
    rx_pos = [0, 0, 0]
    jac = jacobian_code(df, rx_pos)
    assert jac.shape == (3, 4), "output of jacobian_code has wrong shape"
    assert (jac == np.array([[-1, 0, 0, 1], [0, -1, 0, 1], [0, 0, -1, 1]])).all()


def test_cov_mat_identical():
    from src_polars.gnss import obs_covariance_mat

    df = pl.DataFrame({"C_obs_m": [1, 0, 0]})
    cov = obs_covariance_mat(df, "identical")
    assert (cov / cov[0, 0] == np.eye(3)).all()


def test_wls():
    from src_polars.gnss import wls

    # simple LS problem for scalar state
    # result in the computation of the mean of a scalar
    obs = np.array([0.75, 1, 1.25]).reshape(3, 1)
    obs_pred = np.zeros((3, 1))
    jac = np.ones((3, 1))
    cov = np.eye(3)
    x_hat = wls(obs, obs_pred, jac, cov)
    assert x_hat == pytest.approx(np.mean(obs))

    # vector state
    # scalar problem repeated twice
    obs = np.array([0.75, 1, 1.25] * 2).reshape(6, 1)
    obs_pred = np.zeros((6, 1))
    jac = np.repeat([[1, 0], [0, 1]], repeats=3, axis=0)
    cov = np.eye(6)
    x_hat = wls(obs, obs_pred, jac, cov)
    assert x_hat == pytest.approx(np.array([[np.mean(obs[0:3])], [np.mean(obs[3:6])]]))


def test_load_with_elevation_mask():
    from src_polars.prx_tools import load_prx_file

    # single constellation, single frequency
    df_prx = load_prx_file(
        repo_root() / "data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.gz",
        {"G": ["1C"]},
        True,
    )
    assert len(df_prx) == 30205, "default elevation mask value should be 0"

    df_prx = load_prx_file(
        repo_root() / "data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.gz",
        {"G": ["1C"]},
        True,
        10,
    )
    assert len(df_prx) == 24197


def test_cov_mat_elevation():
    from src_polars.gnss import obs_covariance_mat

    df = pl.DataFrame([10, 20, 30], schema=["sat_elevation_deg"])
    cov = obs_covariance_mat(df, "elevation")

    # assert that the 2 matrices are proportional
    assert cov / cov[0, 0] == pytest.approx(
        np.diag(1 / np.sin(np.radians([10, 20, 30])) ** 2) * np.sin(np.radians(10)) ** 2
    )
