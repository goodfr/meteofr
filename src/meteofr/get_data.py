import logging
from functools import cache
from time import sleep
from typing import Any, Optional

import numba  # type: ignore
import numpy as np
import pandas as pd
import requests

# ref: https://portail-api.meteofrance.fr/web/fr/api/test/a5935def-80ae-4e7e-83bc-3ef622f0438d/fe8c79d6-dcae-46f7-9e1f-6d5a8be4c3b8

# courtesy : https://portail-api.meteofrance.fr/web/fr/faq
# Example of a Python implementation for a continuous authentication client.

# unique application id : you can find this in the curl's command to generate jwt token
APPLICATION_ID = (
    "VlZPQjhLQl82eENMblJSZnd3QkRBS1Q0SmhJYTpxTGVwcExKV1ZzRlJtd01YZGVJdlNOMlhKbDhh"
)

# url to obtain acces token
TOKEN_URL = "https://portail-api.meteofrance.fr/token"

# list of french departement
list_dep = [str(i) for i in range(1, 96)]
list_dep.extend(
    [
        "971",
        "972",
        "973",
        "974",
        "975",
        "984",
        "985",
        "986",
        "987",
        "988",
        "99",
    ]
)

# end point API url
url_api = "https://public-api.meteofrance.fr/public/DPClim/v1"

logger = logging.getLogger("meteofr")


class Client:
    """Client class to interact with the API."""

    def __init__(self):
        self.session = requests.Session()

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """_summary_

        Args:
            method (str): http request verb (here: "GET")
            url (str): API end point

        Returns:
            requests.Response: result of request
        """
        # First request will always need to obtain a token first
        if "Authorization" not in self.session.headers:
            self.obtain_token()

        # Optimistically attempt to dispatch reqest
        response = self.session.request(method, url, **kwargs)
        if self.token_has_expired(response):
            # We got an 'Access token expired' response => refresh token
            self.obtain_token()
            # Re-dispatch the request that previously failed
            response = self.session.request(method, url, **kwargs)

        return response

    def token_has_expired(self, response: requests.Response) -> bool:
        """Method to check validity of token.

        Args:
            response (requests.Response): API response.

        Returns:
            bool: True / False answer.
        """
        status = response.status_code
        content_type = response.headers["Content-Type"]
        repJson = response.text
        if status == 401 and "application/json" in content_type:
            if "Invalid JWT token" in repJson["description"]:  # type: ignore
                return True
        return False

    def obtain_token(self):
        # Obtain new token
        data = {"grant_type": "client_credentials"}
        headers = {"Authorization": "Basic " + APPLICATION_ID}
        access_token_response = requests.post(
            TOKEN_URL,
            data=data,
            allow_redirects=False,
            headers=headers,
        )
        token = access_token_response.json()["access_token"]
        # Update session with fresh token
        self.session.headers.update({"Authorization": "Bearer %s" % token})


def get_rqt(request_url: str, format_result: str = "json", error: str = "raise") -> Any:
    """Base function to request API

    Args:
        request_url (str): API end point.
        format_result (str, optional): either ('pd' for pandas.DataFrame, 'json', 'csv' or 'raw'). Defaults to "json".
        error (str, optional): how to handle errors. Defaults to "raise".

    Raises:
        ValueError: raise a value error if  response.status_code not in [200, 300[

    Returns:
        Any: depending on format_result returns a pd.DataFrame, a dict or a requests.Response object.
    """
    from io import StringIO
    from json import loads

    client = Client()
    client.session.headers.update({"Accept": "application/json"})
    response = client.request(
        "GET",
        request_url,
    )

    if response.status_code >= 200 and response.status_code < 300:
        pass
    elif error == "raise":
        raise ValueError(
            f"response.status_code: {response.status_code}\n{response.url}\n{response.text}"
        )
    else:
        logger.warning(
            f"response.status_code: {response.status_code}\n{response.url}\n{response.text}"
        )
        return

    if format_result == "pd":
        df: pd.DataFrame = pd.json_normalize(loads(response.content))
        return df
    elif format_result == "json":
        return loads(response.content)
    elif format_result == "csv":
        return pd.read_csv(StringIO(response.content.decode()), sep=";", decimal=",")
    elif format_result == "raw":
        return response
    else:
        raise ValueError(
            f"format_result: {format_result} must be either pd, json, csv or raw"
        )


@cache
def get_ref(dep: str, prm: str) -> pd.DataFrame:
    """To get data from API into a DataFrame result

    Args:
        dep (int): id of departement
        prm (str): parameter (e.g: temperature)

    Returns:
        pd.DataFrame: result dataframe
    """

    request_url = (
        f"{url_api}/liste-stations/quotidienne?id-departement={dep}&parametre={prm}"
    )

    return get_rqt(request_url=request_url, format_result="pd")


def get_all_ref(list_dep: list[str] = list_dep, use_cache: bool = True) -> pd.DataFrame:
    """To iterate over all departements to fetch station information.

    Args:
        list_dep (list[str], optional): list of french departement. Defaults to list_dep.
        use_cache (bool, optional): to avoid redownloading files if not needed. Defaults to True.

    Returns:
        pd.DataFrame: data framing the required stations list.
    """
    from os import makedirs, path
    from pathlib import Path
    from time import sleep

    from tqdm import tqdm

    logger.debug("begin get all ref")
    dir_cache = Path.home().joinpath(".meteofr")
    cache_file = path.join(dir_cache, "ref.csv")
    makedirs(dir_cache, exist_ok=True)

    if (path.exists(cache_file) is True) and (use_cache is True):
        logger.info("Using cached data for ref.")
        df_ref_geo = pd.read_csv(cache_file)

        if list_dep[-1] in [str(i) for i in df_ref_geo.dep.unique()]:
            return df_ref_geo
        else:
            list_dep = [i for i in list_dep if i not in df_ref_geo.dep.unique()]
            logger.info(f"update list_dep: {list_dep}")

    df_list: list = []
    for i in tqdm(list_dep):
        df_ref_geo = get_ref(dep=i, prm="temperature")
        df_ref_geo["dep"] = i
        if i == list_dep[0]:
            df_ref_geo.to_csv(cache_file, index=False)
        else:
            df_ref_geo.to_csv(cache_file, index=False, header=False, mode="a")
        df_list.append(df_ref_geo)
        sleep(2)

    df_ref_geo = pd.concat(df_list).reset_index(drop=True)

    logger.debug("end get all ref")

    return df_ref_geo


@numba.jit
def hvs(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine for distance computation based on latitude and longitude points.

    Args:
        lat1 (float): latitude of 1st point
        lon1 (float): longitude of 2nd point
        lat2 (float): latitude of 1st point
        lon2 (float): longitude of 2nd point

    Returns:
        float: haversine distance between 2 points (in km)
    """

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    d_km = 6367 * c

    return d_km


@numba.jit
def get_dist(vec: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Find haversine distance matrix between 2 arrays

    Args:
        vec (np.ndarray): array (possibly 1d)
        ref (np.ndarray): array (possibly 1d)

    Returns:
        np.ndarray: array of haversine distance (in km)
    """
    n = vec.shape[0]
    m = ref.shape[0]
    res = np.zeros(shape=(n, m))
    for i in range(n):
        lat1, lon1 = vec[i, :]
        for j in range(m):
            lat2, lon2 = ref[j, :]
            res[i, j] = hvs(lat1, lon1, lat2, lon2)

    return res


def get_closest_n_point(vec: np.ndarray, ref: np.ndarray, n: int = 5) -> np.ndarray:
    """Get the 5 closest station from a given point

    Args:
        vec (np.ndarray): array
        ref (np.ndarray): array

    Returns:
        np.ndarray: array of the index of the top 5 closest points
    """
    mat = get_dist(vec, ref)

    return np.argsort(mat, axis=1)[:, :5]


def get_weather_point(
    dates: list[str],
    point: tuple[float, float],
    df_ref_geo: Optional[pd.DataFrame] = None,
    dest_dir: Optional[str] = "data",
) -> tuple[pd.DataFrame, str]:
    """To fetch weather data per date and position.

    Args:
        dates (list[str]): dates [start, end] for request.
        point (tuple[float, float]): coordinates latitude and longitude
        df_ref_geo (Optional[pd.DataFrame], optional): dataframe of weather station coordinate. Defaults to None (fetch or load from source).
        dest_dir (Optional[str], optional): path to directory to store data. Defaults to "data".

    Returns:
        tuple[pd.DataFrame, str]: returns result (df, station_id) for given point.
    """
    from json import dumps
    from os import makedirs, path

    if dest_dir is not None:
        makedirs(dest_dir, exist_ok=True)

    if df_ref_geo is None:
        # get all stations
        df_ref_geo = get_all_ref(list_dep=list_dep)  # , use_cache=False 106 références

    df_ref_geo = df_ref_geo.convert_dtypes()

    # exclusion des typePoste 5
    # une station de type 5 n'est pas expertisée ou son expertise n'est pas garantie.
    # De plus, la disponibilité des données est occasionnelle
    df_ref_geo = df_ref_geo.loc[df_ref_geo.typePoste != 5]

    # find the closest station from point
    idx = get_closest_n_point(
        vec=np.asarray([point], dtype=(np.float64, np.float64)),
        ref=df_ref_geo[["lat", "lon"]].to_numpy(dtype=(np.float64, np.float64)),
    )

    # --- Find closest & ACTIVE station
    list_station_id = df_ref_geo.iloc[idx.ravel()]["id"].values.tolist()
    list_station_id = [
        f"{i:0>8}" for i in list_station_id
    ]  # padding avec des 0 pour être sur 8 chars

    # pour date de données dispo cf web service /information-station
    # metadonnées des variables disponibles pour la station
    for station_id in list_station_id:
        url_meta = f"https://public-api.meteofrance.fr/public/DPClim/v1/information-station?id-station={station_id}"
        df_meta = get_rqt(url_meta, error="warn")
        if dest_dir is not None:
            with open(path.join(dest_dir, f"meta_{station_id}.json"), "w") as f:
                f.write(dumps(df_meta, indent=4))
        start, end = df_meta[0]["dateDebut"], df_meta[0]["dateFin"]
        end = end if end != "" else "9999-01-01 00:00:00"
        if start <= dates[0] and dates[-1] <= end:
            logger.info(f"station_id: {station_id} compatible with request")
            break
        elif start > dates[0]:
            logger.info(
                f"insufficient data for station_id: {station_id}, start: {start} > {dates[0]}"
            )
        elif end < dates[-1]:
            logger.info(
                f"insufficient data for station_id: {station_id}, end: {end} < {dates[-1]}"
            )
        else:
            logger.info(f"insufficient data for station_id: {station_id}")
        sleep(2)

    # get weather data from closest station
    url_point_weather = f"""https://public-api.meteofrance.fr/public/DPClim/v1/commande-station/quotidienne?id-station={station_id}&date-deb-periode={dates[0]}&date-fin-periode={dates[1]}"""
    logger.info(f"Use url: {url_point_weather}")
    id_rqt = get_rqt(url_point_weather, format_result="json", error="warn")

    # results
    id_cmde = id_rqt["elaboreProduitAvecDemandeResponse"]["return"]
    url_rqt = f"https://public-api.meteofrance.fr/public/DPClim/v1/commande/fichier?id-cmde={id_cmde}"

    df = get_rqt(request_url=url_rqt, error="warn", format_result="csv")

    return df, station_id


def get_weather(
    dates: list[str] | pd.DatetimeIndex,
    point: tuple[float, float],
    dest_dir: str = "data",
    dest_file: Optional[str] = None,
    logger_name: str = "meteofr",
    list_dep: list[str] = list_dep,
) -> pd.DataFrame:
    """User function for downloading data.

    Args:
        dates (list[str] | pd.DatetimeIndex): [start, end] dates for request.
        point (tuple[float, float]): coordinate (latitude, longitude) of point.
        dest_dir (str, optional): path to directory to save data. Defaults to "data".
        dest_file (str, optional): name of the file to save data. Defaults to None.
        logger_name (str, optional): logger name. Defaults to "meteofr".
        list_dep (list[str], optional): list of french departement to get data from. Defaults to list_dep.

    Returns:
        pd.DataFrame: result dataframe
    """
    from itertools import pairwise
    from os import makedirs, path
    from time import sleep

    from tqdm import tqdm

    # donner un site/coord/... + temporalité et récupérer les infos requises
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.debug("Howdy !")

    # --- 1 year max historical request
    dates_dti: pd.DatetimeIndex = pd.DatetimeIndex(dates)
    if (dates_dti[1] - dates_dti[0]).days > 365:
        dates_dti = pd.date_range(start=dates[0], end=dates[1], freq="YE").append(
            dates_dti[[1]]
        )

    time_fmt = "%Y-%m-%dT%H:%M:%SZ"
    dates_ = [i.strftime(time_fmt) for i in dates_dti]

    df_ref_geo = get_all_ref(list_dep=list_dep)  # , use_cache=False

    res = []
    for i, j in tqdm(pairwise(dates_)):
        doc, station_id = get_weather_point(
            dates=[i, j],
            point=point,
            df_ref_geo=df_ref_geo,
        )
        res.append(doc)
        sleep(2)

    makedirs(dest_dir, exist_ok=True)
    dest_file_: str = (
        f"df_res_{station_id}_{'_'.join(dates_dti.strftime('%Y%m%d'))}.csv"
        if dest_file is None
        else str(dest_file)
    )
    df = pd.concat(res)

    df.to_csv(
        path.join(
            dest_dir,
            dest_file_,
        ),
        index=False,
    )

    return df


if __name__ == "__main__":
    # --- simple test
    # test_point = (45.932050, 2.000847)
    test_point = (47.218102, -1.552800)

    td = pd.Timestamp("today", tz="Europe/Paris").normalize().tz_convert("UTC")
    dates = pd.DatetimeIndex([td - pd.Timedelta("30d"), td])  # 1 year max

    df = get_weather(dates=dates, point=test_point)
    print(f"df shape: {df.shape}")
