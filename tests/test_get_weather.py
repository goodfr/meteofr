"""Unit testing"""


def test_get_weather(test_point=(47.218102, -1.552800)):
    """To test basic download"""
    from pandas import Timestamp, Timedelta, DatetimeIndex
    from src.meteofr.get_data import get_weather

    td = Timestamp("today", tz="Europe/Paris").normalize().tz_convert("UTC")
    dates = DatetimeIndex([td - Timedelta("30d"), td])  # 1 an max

    df = get_weather(dates=dates, point=test_point)

    assert df.shape[0] > 0, "Test data can not be empty."


def test_hard_point():
    """To test harder point to match to station"""
    test_point = (50.7237, 2.88079)  # 10 neighboors needed
    test_get_weather(test_point=test_point)


def test_get_many():
    import pandas as pd

    # df = pd.read_csv("data/df_geo_points.csv")
    # df.iloc[:10].to_dict(orient="tight").__repr__()
    # df.iloc[:10].to_dict(orient="split").__repr__()
    # pd.DataFrame({'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'columns': ['latitude_site', 'longitude_site', 'altitude_site'], 'data': [[41.9333, 9.33333, 360], [42.4467, 3.1659, 71], [42.4476, 3.1646, 73], [42.4717, 2.96754, 914], [42.4895, 2.72684, 284], [42.5017, 3.12207, 6], [42.5238, 2.84637, 77], [42.5533, 3.04276, 3], [42.5796, 2.86658, 69], [42.6199, 2.75325, 126]], 'index_names': [None], 'column_names': [None]})
    # pd.DataFrame({'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'columns': ['latitude_site', 'longitude_site', 'altitude_site'], 'data': [[41.9333, 9.33333, 360], [42.4467, 3.1659, 71], [42.4476, 3.1646, 73], [42.4717, 2.96754, 914], [42.4895, 2.72684, 284], [42.5017, 3.12207, 6], [42.5238, 2.84637, 77], [42.5533, 3.04276, 3], [42.5796, 2.86658, 69], [42.6199, 2.75325, 126]]})

    # df.iloc[:10].to_dict(orient="index").__repr__()
    # df.iloc[:10].to_dict(orient="records").__repr__()
    # pd.DataFrame({0: {'latitude_site': 41.9333, 'longitude_site': 9.33333, 'altitude_site': 360}, 1: {'latitude_site': 42.4467, 'longitude_site': 3.1659, 'altitude_site': 71}, 2: {'latitude_site': 42.4476, 'longitude_site': 3.1646, 'altitude_site': 73}, 3: {'latitude_site': 42.4717, 'longitude_site': 2.96754, 'altitude_site': 914}, 4: {'latitude_site': 42.4895, 'longitude_site': 2.72684, 'altitude_site': 284}, 5: {'latitude_site': 42.5017, 'longitude_site': 3.12207, 'altitude_site': 6}, 6: {'latitude_site': 42.5238, 'longitude_site': 2.84637, 'altitude_site': 77}, 7: {'latitude_site': 42.5533, 'longitude_site': 3.04276, 'altitude_site': 3}, 8: {'latitude_site': 42.5796, 'longitude_site': 2.86658, 'altitude_site': 69}, 9: {'latitude_site': 42.6199, 'longitude_site': 2.75325, 'altitude_site': 126}})

    df = pd.DataFrame(
        [
            {"latitude_site": 41.9333, "longitude_site": 9.33333, "altitude_site": 360},
            {"latitude_site": 42.4467, "longitude_site": 3.1659, "altitude_site": 71},
            {"latitude_site": 42.4476, "longitude_site": 3.1646, "altitude_site": 73},
            {"latitude_site": 42.4717, "longitude_site": 2.96754, "altitude_site": 914},
            {"latitude_site": 42.4895, "longitude_site": 2.72684, "altitude_site": 284},
            {"latitude_site": 42.5017, "longitude_site": 3.12207, "altitude_site": 6},
            {"latitude_site": 42.5238, "longitude_site": 2.84637, "altitude_site": 77},
            # {
            #     "latitude_site": 42.5533,
            #     "longitude_site": 3.04276,
            #     "altitude_site": 3,
            # },  # pb here
            {"latitude_site": 42.5796, "longitude_site": 2.86658, "altitude_site": 69},
            {"latitude_site": 42.6199, "longitude_site": 2.75325, "altitude_site": 126},
        ]
    )

    for row in df.itertuples():
        test_get_weather(test_point=(row.latitude_site, row.longitude_site))
