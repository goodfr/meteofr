[![PyPI - Version](https://img.shields.io/pypi/v/meteofr.svg)](https://pypi.org/project/meteofr)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/meteofr.svg)](https://pypi.org/project/meteofr)

-----

Tool to fetch weather data from Meteo France API based on latitude and longitude coordinates.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [License](#license)
- [See also](#see-also)

## Installation

```console
pip install meteofr
```

## Quickstart

Set up a (free) API key at [portail-api.meteofrance.fr](https://portail-api.meteofrance.fr/web/fr/api/DonneesPubliquesClimatologie).

A first download of the list of weather stations is required to compute closest active at the time station from given point.

```python
from pandas import Timestamp, Timedelta, DatetimeIndex
from meteofr.get_data import get_weather

test_point = (47.218102, -1.552800)

td = Timestamp("today", tz="Europe/Paris").normalize().tz_convert("UTC")
dates = DatetimeIndex([td - Timedelta("30d"), td])  # 1 year max

df = get_weather(dates=dates, point=test_point)
print(df.head)

#       POSTE      DATE   RR  QRR  DRR  QDRR   TN  QTN  HTN  ...  QTMERMAX  TMERMIN  QTMERMIN  HNEIGEF  QHNEIGEF  NEIGETOTX  QNEIGETOTX  NEIGETOT06  QNEIGETOT06  
# 0  44109012  20250301  0.0    1  NaN   NaN  2.8    1  448  ...       NaN      NaN       NaN      NaN       NaN        NaN         NaN         NaN          NaN  
# 1  44109012  20250302  0.0    1  NaN   NaN  1.8    1  442  ...       NaN      NaN       NaN      NaN       NaN        NaN         NaN         NaN          NaN  
# 2  44109012  20250303  0.0    1  NaN   NaN  2.3    1  553  ...       NaN      NaN       NaN      NaN       NaN        NaN         NaN         NaN          NaN  
# 3  44109012  20250304  0.0    1  NaN   NaN  4.9    1  547  ...       NaN      NaN       NaN      NaN       NaN        NaN         NaN         NaN          NaN  
# 4  44109012  20250305  0.2    1  NaN   NaN  4.0    1  647  ...       NaN      NaN       NaN      NaN       NaN        NaN         NaN         NaN          NaN  

# [5 rows x 136 columns]
```

## License

`meteofr` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## See also

- meteofetch ([data.gouv](https://www.data.gouv.fr/fr/reuses/meteofetch/) and [github](https://github.com/CyrilJl/MeteoFetch) links) for Arome model forecasting
- [meteostat](https://dev.meteostat.net/) for global weather data (based on [data.europa](.eu/data/datasets) for Europe data)
