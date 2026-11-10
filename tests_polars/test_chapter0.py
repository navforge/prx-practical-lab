import polars as pl

from src_polars.helpers import repo_root
from src_polars.prx_tools import load_prx_file, apply_iono_corr, extract_col

TLSE_FILE = repo_root() / "data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.gz"


def test_load_prx_file():
    # single constellation, single frequency
    df_prx = load_prx_file(TLSE_FILE, {"G": ["1C"]})
    assert set(df_prx["constellation"].unique()) == {"G"}, (
        "DataFrame should contain only GPS observations"
    )
    assert set(df_prx["rnx_obs_identifier"].unique()) == {"1C"}, (
        "DataFrame should contain only 1C observations"
    )
    assert len(df_prx) == 30205

    # dual constellation, single frequency
    df_prx = load_prx_file(TLSE_FILE, {"G": ["1C"], "E": ["1X"]})
    assert set(df_prx["constellation"].unique()) == {"G", "E"}, (
        "DataFrame should contain GPS and Galileo observations"
    )
    assert set(
        df_prx.filter(pl.col("constellation") == "G")["rnx_obs_identifier"].unique()
    ) == {"1C"}, "DataFrame should contain only 1C observations for GPS"
    assert set(
        df_prx.filter(pl.col("constellation") == "E")["rnx_obs_identifier"].unique()
    ) == {"1X"}, "DataFrame should contain only 1X observations for Galileo"
    assert len(df_prx) == 53071

    # dual constellation, dual frequency
    df_prx = load_prx_file(TLSE_FILE, {"G": ["1C", "2X"], "E": ["1X", "5X"]})
    assert set(df_prx["constellation"].unique()) == {"G", "E"}, (
        "DataFrame should contain GPS and Galileo observations"
    )
    assert set(
        df_prx.filter(pl.col("constellation") == "G")["rnx_obs_identifier"].unique()
    ) == {"1C", "2X"}, "DataFrame should contain 1C and 2X observations for GPS"
    assert set(
        df_prx.filter(pl.col("constellation") == "E")["rnx_obs_identifier"].unique()
    ) == {"1X", "5X"}, "DataFrame should contain 1X and 5X observations for Galileo"
    assert len(df_prx) == 99181


def test_iono():
    df_prx = pl.DataFrame(
        {
            "C_obs_m": [20_000_001, 21_000_002, 22_000_003],
            "iono_delay_m": [1, 2, 3],
        }
    )
    df_prx = apply_iono_corr(df_prx)

    assert "C_obs_corr_m" in df_prx.columns, "No column named 'C_obs_corr_m' found"
    assert df_prx["C_obs_corr_m"].to_list() == [20_000_000, 21_000_000, 22_000_000], (
        "Wrong value of the iono-corrected code observation found in the first 3 rows."
    )


def test_extract_col():
    df_prx = load_prx_file(TLSE_FILE, {"G": ["1C"]}, True)
    df_reduced = extract_col(df_prx)
    assert set(df_reduced.columns) == {
        "epoch",
        "sat_clock_offset_m",
        "sat_pos_x_m",
        "sat_pos_y_m",
        "sat_pos_z_m",
        "C_obs_m",
        "constellation",
        "prn",
    }
