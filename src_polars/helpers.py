from pathlib import Path

import polars as pl
import numpy as np
import plotly.express as px
import plotly.offline as pyo
from plotly.subplots import make_subplots

EARTH_FLATTENINGFACTOR = 1 / 298.257223563
EARTH_SEMIMAJORAXIS_M = 6378137.0
EARTH_ECCENTRICITY = np.sqrt(2 * EARTH_FLATTENINGFACTOR - EARTH_FLATTENINGFACTOR**2)


def repo_root():
    return Path(__file__).parents[1]


def ecef_to_geodetic(
    pos_ecef_x: np.ndarray, pos_ecef_y: np.ndarray, pos_ecef_z: np.ndarray
):
    """
    pos_ecef_{x,y,z}: np.array of shape (n,)
    Reference:
    GNSS data Processing, Vol. I: Fundamentals and Algorithms. Equations (B.4),(B.5),(B.6)
    """
    p = np.linalg.norm(np.array([pos_ecef_x, pos_ecef_y]), axis=0)
    longitude_rad = np.arctan2(pos_ecef_y, pos_ecef_x)
    precision_rad = np.full(
        pos_ecef_x.shape, 1.6e-10
    )  # desired precision in radians, corresponds to 1 mm
    delta_phi_rad = np.full(
        pos_ecef_x.shape, 1.0
    )  # initialization to a value larger than precision
    altitude_m = np.zeros(pos_ecef_x.shape)
    latitude_rad = np.arctan2(pos_ecef_z, p * (1 - EARTH_ECCENTRICITY**2))
    while (delta_phi_rad > precision_rad).any():
        n = EARTH_SEMIMAJORAXIS_M / np.sqrt(
            1 - EARTH_ECCENTRICITY**2 * np.sin(latitude_rad) ** 2
        )
        # altitude_previous = altitude_m
        altitude_m = p / np.cos(latitude_rad) - n
        # delta_h_m = np.abs(altitude_m - altitude_previous)
        latitude_prev = latitude_rad
        latitude_rad = np.arctan2(
            pos_ecef_z,
            p * (1 - n * EARTH_ECCENTRICITY**2 / (n + altitude_m)),
        )
        delta_phi_rad = np.abs(latitude_rad - latitude_prev)
    return latitude_rad, longitude_rad, altitude_m


def ecef_to_enu_matrix(lat_rad, lon_rad):
    # ESA Book - eq (B.11)
    R = np.array(
        [
            [-np.sin(lon_rad), np.cos(lon_rad), 0],
            [
                -np.cos(lon_rad) * np.sin(lat_rad),
                -np.sin(lon_rad) * np.sin(lat_rad),
                np.cos(lat_rad),
            ],
            [
                np.cos(lon_rad) * np.cos(lat_rad),
                np.sin(lon_rad) * np.cos(lat_rad),
                np.sin(lat_rad),
            ],
        ]
    )
    return R


def ecef_to_enu(xyz, xyz_ref):
    (lat_rad, lon_rad, _) = ecef_to_geodetic(
        xyz_ref[0],
        xyz_ref[1],
        xyz_ref[2],
    )
    R_ecef_to_enu = ecef_to_enu_matrix(lat_rad, lon_rad)
    return R_ecef_to_enu @ (xyz - xyz_ref)


def compute_enu_pos_error(results_df: pl.DataFrame, ref_pos: list):
    """
    Compute the position error in the ENU frame, add the ENU position error to the results_df as new columns.

    Inputs:
    - results_df: pl.DataFrame, dataframe containing the columns "epochs", "pos_x", "pos_y", "pos_z"
    - ref_pos: list, true position of the receiver in ECEF frame

    Outputs:
    - results_df: same pl.Dataframe with additional columns "error_e", "error_n", "error_u", "error_hor", "error_3d"
    """
    n_epoch = len(results_df)
    enu_est = np.empty((n_epoch, 3))
    # TODO: vectorize this loop
    for idx_epoch in range(n_epoch):
        enu_est[idx_epoch, :] = ecef_to_enu(
            np.array(results_df.select(["pos_x", "pos_y", "pos_z"]).row(idx_epoch)),
            ref_pos,
        )
    return results_df.hstack(
        pl.DataFrame(enu_est, schema=["error_e", "error_n", "error_u"]),
    ).with_columns(
        error_hor=(pl.col("error_e") ** 2 + pl.col("error_n") ** 2).sqrt(),
        error_3d=(
            pl.col("error_e") ** 2 + pl.col("error_n") ** 2 + pl.col("error_u") ** 2
        ).sqrt(),
    )


def plot_enu_error(
    filepath_save: str,
    results_df: pl.DataFrame,
    title: str,
):
    """
    Plot ENU error using Plotly Express.
    One subplot for horizontal error scatter plot, another for vertical error time series.

    Inputs:
    - filepath_save: str, path to save the HTML file
    - results_df: pl.DataFrame with columns "epoch", "error_e", "pos_n", "pos_u"
    - title: str, plot title
    """
    # Create horizontal error scatter plot
    fig_hor = px.scatter(
        results_df,
        x="error_e",
        y="error_n",
        title="Horizontal Position Error",
        labels={"error_e": "East error [m]", "error_n": "North error [m]"},
    )
    fig_hor.update_traces(mode="lines+markers", marker=dict(size=4), name=title)

    # Create vertical error time series plot
    fig_vert = px.scatter(
        results_df,
        x="epoch",
        y="error_u",
        title="Vertical Position Error",
        labels={"epoch": "Epoch", "error_u": "Up error [m]"},
    )
    fig_vert.update_traces(mode="lines+markers", marker=dict(size=4))

    # Combine subplots
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Horizontal Error", "Vertical Error")
    )
    fig.add_trace(fig_hor.data[0], row=1, col=1)
    fig.add_trace(fig_vert.data[0], row=1, col=2)

    fig.update_xaxes(title_text="East error [m]", row=1, col=1)
    fig.update_yaxes(title_text="North error [m]", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Up error [m]", row=1, col=2)

    fig.update_layout(title_text=title, xaxis=dict(scaleanchor="y", scaleratio=1))
    pyo.plot(fig, auto_open=False, filename=filepath_save + ".html")


def plot_enu_error_cdf(filepath_save: str, results_df: pl.DataFrame, title: str):
    """
    Save a figure of the ENU error cumulative density function. One subplot for the horizontal error, another for the vertical error.

    Inputs:
    - filepath_save: str, location for saving the figure. Ex: "figures/enu.png"
    - results_df: pl.DataFrame, dataframe containing the columns "epochs", "error_e", "error_n", "error_u", "error_hor"
    - title: str, plot title
    """
    # Create ECDF plots
    fig_hor = px.ecdf(
        results_df,
        x="error_hor",
        title="Horizontal Position Error CDF",
        labels={"error_hor": "Horizontal error [m]"},
    )
    fig_vert = px.ecdf(
        results_df,
        x="error_u",
        title="Vertical Position Error CDF",
        labels={"error_u": "Up error [m]"},
    )

    # Customize layout
    fig_hor.update_layout(showlegend=True)
    fig_vert.update_layout(showlegend=True)

    # Combine subplots
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(fig_hor.data[0], row=1, col=1)
    fig.add_trace(fig_vert.data[0], row=1, col=2)

    fig.update_xaxes(title_text="Horizontal error [m]", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative distribution", row=1, col=1)
    fig.update_xaxes(title_text="Up error [m]", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative distribution", row=1, col=2)

    fig.update_layout(title_text=title)
    pyo.plot(fig, auto_open=False, filename=filepath_save + ".html")


def analyze_results_feather(file):
    results = pl.read_ipc(file)
    analyze_results(results)


def analyze_results(results: pl.DataFrame):
    results_desc = results.select(
        ["error_e", "error_n", "error_u", "error_hor", "error_3d"]
    ).describe(percentiles=[0.25, 0.5, 0.75, 0.95])
    q_50 = results_desc.filter(pl.col("statistic") == "50%").item(0, "error_3d")
    q_95 = results_desc.filter(pl.col("statistic") == "95%").item(0, "error_3d")
    score = 0.5 * (q_50 + q_95)

    print(results_desc)
    print("------------------------")
    print(f"Score = 0.5 * ({q_50:.3f} + {q_95:.3f})")
    print(f"      = {score:.3f}")
    return score


def find_clean_intervals(df: pl.DataFrame, n_intervals: int = 5):
    # compute nb of cycle slips per epochs
    n_cycle_slips = (
        df.select(["epoch", "LLI"])
        .group_by("epoch", maintain_order=True)
        .agg(pl.col("LLI").sum().alias("n_lli"))
    )
    # compute cumulative number of cycle slips
    n_cycle_slips = n_cycle_slips.with_columns(
        pl.col("n_lli").cum_sum().alias("cumulative_lli")
    )

    # compute start and end date of periods without LLI
    continuous_periods = n_cycle_slips.group_by("cumulative_lli").agg(
        pl.col("epoch").first().alias("start"),
        pl.col("epoch").last().alias("end"),
    )
    continuous_periods = continuous_periods.with_columns(
        (pl.col("end") - pl.col("start")).alias("duration")
    )
    # return 5 longest periods
    return continuous_periods.sort(by="duration", descending=True).head(n_intervals)
