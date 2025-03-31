# meteofr

[![PyPI - Version](https://img.shields.io/pypi/v/meteofr.svg)](https://pypi.org/project/meteofr)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/meteofr.svg)](https://pypi.org/project/meteofr)

-----

Tool to fetch weather data from Meteo France API based on latitude and longitude and longitude coordinates.

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install meteofr
```

## Quickstart

A first download of the list of weather stations is required to compute closest active station from given point.

```python
from src.meteofr.get_data import get_weather

test_point = (47.218102, -1.552800)

td = Timestamp("today", tz="Europe/Paris").normalize().tz_convert("UTC")
dates = DatetimeIndex([td - Timedelta("30d"), td])  # 1 year max per request

df = get_weather(dates=dates, point=test_point)
```

## License

`meteofr` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
