"""Unit testing"""


def test_get_weather():
    """To test basic download"""
    from pandas import Timestamp, Timedelta, DatetimeIndex
    from src.meteofr.get_data import get_weather

    test_point = (47.218102, -1.552800)

    td = Timestamp("today", tz="Europe/Paris").normalize().tz_convert("UTC")
    dates = DatetimeIndex([td - Timedelta("30d"), td])  # 1 an max

    df = get_weather(dates=dates, point=test_point)

    assert df.shape[0] > 0, "Test data can not be empty."


def test_pip_install():
    """To test import after pip install"""
    from pandas import Timestamp, Timedelta, DatetimeIndex
    from meteofr.get_data import get_weather  # type: ignore # TODO: add package stubs

    test_point = (47.218102, -1.552800)

    td = Timestamp("today", tz="Europe/Paris").normalize().tz_convert("UTC")
    dates = DatetimeIndex([td - Timedelta("30d"), td])  # 1 an max

    df = get_weather(dates=dates, point=test_point)

    assert df.shape[0] > 0, "Test data can not be empty."
