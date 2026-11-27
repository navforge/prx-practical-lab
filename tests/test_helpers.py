from src.helpers import lat_lon_hash


def test_lat_lon_hash() -> None:
    # Given a latitude and longitude with 6 significant decimals
    lat = 56.789123
    lon = 1.234567
    # When computing the hash (with a small time cost to speed up the test)
    hash_1 = lat_lon_hash(lat_deg=lat, lon_deg=lon, time_cost=1)
    # And when then computing the hash with a change in the 7th decimal, we expect it to be the same
    assert hash_1 == lat_lon_hash(lat_deg=lat + 1e-7, lon_deg=lon - 1e-7, time_cost=1)
    # Then when applying a change in the 6th decimal we expect a different hash
    assert hash_1 != lat_lon_hash(lat_deg=lat + 1e-6, lon_deg=lon, time_cost=1)
    assert hash_1 != lat_lon_hash(lat_deg=lat - 1e-6, lon_deg=lon, time_cost=1)
    assert hash_1 != lat_lon_hash(lat_deg=lat, lon_deg=lon + 1e-6, time_cost=1)
    assert hash_1 != lat_lon_hash(lat_deg=lat, lon_deg=lon - 1e-6, time_cost=1)
    # When using a different time cost parameter we expect the has to change
    assert hash_1 != lat_lon_hash(lat_deg=lat, lon_deg=lon, time_cost=2)
