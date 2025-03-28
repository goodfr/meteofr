import numpy as np
import pandas as pd
import requests
import numba  # type: ignore
from functools import cache
import logging
from typing import Any, Optional
from time import sleep

# ref: https://portail-api.meteofrance.fr/web/fr/api/test/a5935def-80ae-4e7e-83bc-3ef622f0438d/fe8c79d6-dcae-46f7-9e1f-6d5a8be4c3b8

# courtesy : https://portail-api.meteofrance.fr/web/fr/faq
# Example of a Python implementation for a continuous authentication client.
# It's necessary to :
# - update APPLICATION_ID
# - update request_url at the end of the script

# unique application id : you can find this in the curl's command to generate jwt token
APPLICATION_ID = (
    "VlZPQjhLQl82eENMblJSZnd3QkRBS1Q0SmhJYTpxTGVwcExKV1ZzRlJtd01YZGVJdlNOMlhKbDhh"
)

# url to obtain acces token
TOKEN_URL = "https://portail-api.meteofrance.fr/token"

list_dep = [str(i) for i in range(1, 96)]
list_dep.extend(
    ["971", "972", "973", "974", "975", "984", "985", "986", "987", "988", "99"]
)
url_api = "https://public-api.meteofrance.fr/public/DPClim/v1"

logger = logging.getLogger("meteofr")


class Client(object):
    def __init__(self):
        self.session = requests.Session()

    def request(self, method, url, **kwargs):
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

    def token_has_expired(self, response):
        status = response.status_code
        content_type = response.headers["Content-Type"]
        repJson = response.text
        if status == 401 and "application/json" in content_type:
            repJson = response.text
            if "Invalid JWT token" in repJson["description"]:
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
            headers=headers,  # , verify=False,
        )
        token = access_token_response.json()["access_token"]
        # Update session with fresh token
        self.session.headers.update({"Authorization": "Bearer %s" % token})


def get_rqt(request_url: str, format_result: str = "json", error: str = "raise") -> Any:
    """Base function to request API

    Args:
        request_url (str): _description_
        format_result (str, optional): either ('pd' for pandas.DataFrame, 'json', 'csv' or 'raw'). Defaults to "json".
        error (str, optional): _description_. Defaults to "raise".

    Raises:
        ValueError: _description_

    Returns:
        Any: _description_
    """
    from json import loads
    from io import StringIO

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

    # logger.debug(f"begin get ref: {dep}, {prm}")
    request_url = (
        f"{url_api}/liste-stations/quotidienne?id-departement={dep}&parametre={prm}"
    )

    return get_rqt(request_url=request_url, format_result="pd")


# @cache
# def get_all_ref(*list_dep: str) -> pd.DataFrame: ## cache nécessite des args bytable (non list)
def get_all_ref(list_dep: list[str] = list_dep, use_cache: bool = True) -> pd.DataFrame:
    from pathlib import Path
    from os import makedirs, path
    from tqdm import tqdm
    from time import sleep

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
    """Haversine for distance computation."""

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    d_km = 6367 * c

    return d_km


@numba.jit
def get_dist(vec: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Find haversine distance matrix."""
    n = vec.shape[0]
    m = ref.shape[0]
    res = np.zeros(shape=(n, m))
    for i in range(n):
        lat1, lon1 = vec[i, :]
        for j in range(m):
            lat2, lon2 = ref[j, :]
            res[i, j] = hvs(lat1, lon1, lat2, lon2)

    return res


def get_closest_n_point(
    vec: np.ndarray, ref: np.ndarray, n: int = 5
) -> np.ndarray:  # tuple[np.ndarray, np.ndarray]
    """Get closest station from point.

    Args:
        vec (np.ndarray): _description_
        ref (np.ndarray): _description_

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    mat = get_dist(vec, ref)

    # return mat.argmin(axis=1), mat.min(axis=1)
    # ii = np.argsort(mat, axis=1)[:,:5]
    # mat[ii[0:5,:]]
    # ii.ravel()
    # mat[:,ii.ravel()]
    # mat[:,]
    # np.sort(mat, axis=1)[:5]
    return np.argsort(mat, axis=1)[:, :5]


def get_weather_point(
    dates: list[str],
    point: Optional[tuple[float, float]] = None,
    id: Optional[str] = None,
    name: Optional[str] = None,
    df_ref_geo: Optional[pd.DataFrame] = None,
    dest_dir: Optional[str] = "data",
):
    """To fetch weather data per date and position.

    Args:
        dates (list[str, str]): _description_
        point (_type_, optional): _description_. Defaults to None.
        id (_type_, optional): _description_. Defaults to None.
        name (_type_, optional): _description_. Defaults to None.

    Raises:
        NotImplementedError: _description_
    """
    from os import makedirs, path
    from json import dumps
    # from pathlib import Path

    assert not all([point is None, id is None, name is None]), (
        "1 parameter point, id or name must be given"
    )

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

    # find the closest
    if id or name:
        raise NotImplementedError
    elif point is not None:
        # idx, d = get_closest_n_point(
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
                f"data insufficient for station_id: {station_id}, start: {start} > {dates[0]}"
            )
        elif end < dates[-1]:
            logger.info(
                f"data insufficient for station_id: {station_id}, end: {end} < {dates[-1]}"
            )
        else:
            logger.info(f"data insufficient for station_id: {station_id}")
        sleep(2)

    # get weather data from closest station
    url_point_weather = f"""https://public-api.meteofrance.fr/public/DPClim/v1/commande-station/quotidienne?id-station={station_id}&date-deb-periode={dates[0]}&date-fin-periode={dates[1]}"""
    logger.info(f"Use url: {url_point_weather}")
    id_rqt = get_rqt(url_point_weather, format_result="json", error="warn")

    # results
    id_cmde = id_rqt["elaboreProduitAvecDemandeResponse"]["return"]
    url_rqt = f"https://public-api.meteofrance.fr/public/DPClim/v1/commande/fichier?id-cmde={id_cmde}"

    doc = get_rqt(request_url=url_rqt, error="warn", format_result="csv")

    return doc, station_id


def get_weather(
    dates: list[str] | pd.DatetimeIndex,
    point: tuple[float, float],
    dest_dir="data",
    logger_name: str = "meteofr",
    list_dep: list[str] = list_dep,
) -> pd.DataFrame:
    """_summary_

    Args:
        dates (list[str] | pd.DatetimeIndex): _description_
        point (tuple[float, float]): _description_
        dest_dir (str, optional): _description_. Defaults to "data".
        logger_name (str, optional): _description_. Defaults to "meteofr".
        list_dep (list[str], optional): _description_. Defaults to list_dep.

    Raises:
        TypeError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    from itertools import pairwise
    from time import sleep
    from tqdm import tqdm
    from os import makedirs, path

    # donner un site/coord/... + temporalité et récupérer les infos requises
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Howdy !")

    if isinstance(dates, pd.DatetimeIndex):
        time_fmt = "%Y-%m-%dT%H:%M:%SZ"
        dates_ = [i.strftime(time_fmt) for i in dates]
    elif isinstance(dates[0], str):
        dates_ = dates
    else:
        raise TypeError

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
    df = pd.concat(res)

    df.to_csv(path.join(dest_dir, f"df_res_{station_id}.csv"), index=False)

    return df


if __name__ == "__main__":
    # --- test simple
    # test_point = (45.932050, 2.000847)
    test_point = (47.218102, -1.552800)

    # dates = [start.strftime(time_fmt), td.strftime(time_fmt)]
    # get_weather(dates=dates, point=test_point)

    # dates = pd.DatetimeIndex([td - pd.Timedelta("370d"), td])  # 1 an max
    td = pd.Timestamp("today", tz="Europe/Paris").normalize().tz_convert("UTC")
    dates = pd.DatetimeIndex([td - pd.Timedelta("30d"), td])  # 1 an max
    if (dates[1] - dates[0]).days > 365:
        dates = pd.date_range(start=dates[0], end=dates[1], freq="YE").append(
            dates[[1]]
        )

    df = get_weather(dates=dates, point=test_point)

    ""

    # TODO : se baser sur station la plus proche AVEC données publique (ou encore d'actualité...)
