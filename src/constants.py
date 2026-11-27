import src.helpers as helpers
import numpy as np

C_LIGHT_MPS = 299_792_458


def add_antenna_height(rx_ecef, ant_enu):
    lat, lon, _ = helpers.ecef_to_geodetic(
        np.array([rx_ecef[0]]), np.array([rx_ecef[1]]), np.array([rx_ecef[2]])
    )
    rot_mat = helpers.ecef_to_enu_matrix(lat[0], lon[0])
    rx_ecef += rot_mat.T @ np.array(ant_enu)
    return list(rx_ecef)


# from IGS0OPSSNX_20240010000_01D_01D_SOL.SNX
tlse_ecef = [4.62785160311780e06, 1.19640383149059e05, 4.37299376985211e06]
eccentricity_enu = [0, 0, 1.053]
TLSE_2024001_ECEF = add_antenna_height(tlse_ecef, eccentricity_enu)

# from IGS0OPSSNX_20240010000_01D_01D_SOL.SNX
tlsg_ecef = [4.62868464005436e06, 1.19997351897216e05, 4.37211046833398e06]
eccentricity_enu = [0, 0, 0.4410]
TLSG_2024001_ECEF = add_antenna_height(tlsg_ecef, eccentricity_enu)
