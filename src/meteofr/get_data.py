import numpy as np
import pandas as pd
import requests
import numba  # type: ignore
from functools import cache
import logging
from typing import Any, Optional

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


def get_closest_point(
    vec: np.ndarray, ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Get closest station from point.

    Args:
        vec (np.ndarray): _description_
        ref (np.ndarray): _description_

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    mat = get_dist(vec, ref)
    return mat.argmin(axis=1), mat.min(axis=1)


def get_weather(
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

    # find the closest
    if id or name:
        raise NotImplementedError
    elif point is not None:
        d, idx = get_closest_point(
            vec=np.asarray([point], dtype=(np.float64, np.float64)),
            ref=df_ref_geo[["lat", "lon"]].to_numpy(dtype=(np.float64, np.float64)),
        )

    # mat = get_dist(
    #     vec=np.asarray([point], dtype=(np.float64, np.float64)),
    #     ref=df_ref_geo[["lat", "lon"]].to_numpy(dtype=(np.float64, np.float64)),
    # )

    # df_ref_geo.iloc[idx]
    # df_ref_geo[["lat", "lon"]].apply(
    #     lambda x: np.abs(x["lat"] - point[0] + x["lon"] - point[1]), axis=1
    # ).sort_values()
    # df_ref_geo[["lat", "lon"]].iloc[0].lat

    # df_ref_geo["l1_dist"] = df_ref_geo[["lat", "lon"]].apply(
    #     lambda x: np.abs(x["lat"] - point[0] + x["lon"] - point[1]), axis=1
    # )
    # df_ref_geo.sort_values(by="l1_dist")
    # df_ref_geo["l2_dist"] = df_ref_geo[["lat", "lon"]].apply(
    #     lambda x: np.sqrt((x["lat"] - point[0]) ** 2 + (x["lon"] - point[1]) ** 2),
    #     axis=1,
    # )
    # df_ref_geo.sort_values(by="l2_dist")
    # df_ref_geo["hvs_dist"] = df_ref_geo[["lat", "lon"]].apply(
    #     lambda x: hvs(x["lat"], point[0], x["lon"], point[1]), axis=1
    # )
    # df_ref_geo.sort_values(by="hvs_dist")

    # closest station
    # station_id = "02035001"
    station_id = df_ref_geo.iloc[idx][["id"]].values[0][0]
    station_id = f"{station_id:0>8}"  # padding avec des 0 pour être sur 8 chars

    # metadonnées des variables disponibles pour la station
    if dest_dir is not None:
        url_meta = f"https://public-api.meteofrance.fr/public/DPClim/v1/information-station?id-station={station_id}"
        df_meta = get_rqt(url_meta, error="warn")
        with open(path.join(dest_dir, f"meta_{station_id}.json"), "w") as f:
            f.write(dumps(df_meta))

    # get weather data from closest station
    # url_point_weather = """https://public-api.meteofrance.fr/public/DPClim/v1/commande-station/quotidienne?id-station=02035001&date-deb-periode=2025-01-01T00%3A00%3A00Z&date-fin-periode=2025-01-10T00%3A00%3A00Z"""
    url_point_weather = f"""https://public-api.meteofrance.fr/public/DPClim/v1/commande-station/quotidienne?id-station={station_id}&date-deb-periode={dates[0]}&date-fin-periode={dates[1]}"""
    logger.info(f"Use url: {url_point_weather}")
    id_rqt = get_rqt(url_point_weather, format_result="json", error="warn")

    # results
    id_cmde = id_rqt["elaboreProduitAvecDemandeResponse"]["return"]
    url_rqt = f"https://public-api.meteofrance.fr/public/DPClim/v1/commande/fichier?id-cmde={id_cmde}"

    doc = get_rqt(request_url=url_rqt, error="warn", format_result="csv")

    return doc, station_id


if __name__ == "__main__":
    from itertools import pairwise
    from time import sleep
    from tqdm import tqdm
    from os import makedirs, path

    # on veut
    # donner un site/coord/... + temporalité et récupérer les infos requises
    logger = logging.getLogger("meteofr")
    logger.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    logger.info("Howdy !")

    # test_point = (45.932050, 2.000847)
    test_point = (47.218102, -1.552800)

    td = pd.Timestamp("today", tz="Europe/Paris").normalize().tz_convert("UTC")
    start = td - pd.Timedelta("1d")
    time_fmt = "%Y-%m-%dT%H:%M:%SZ"

    # --- test simple
    # dates = [start.strftime(time_fmt), td.strftime(time_fmt)]
    # get_weather(dates=dates, point=test_point)

    dates = pd.date_range(start=td - pd.Timedelta("30d"), end=td, freq="10d")

    df_ref_geo = get_all_ref(list_dep=list_dep)  # , use_cache=False

    res = []
    for i, j in tqdm(pairwise(dates)):
        doc, station_id = get_weather(
            dates=[i.strftime(time_fmt), j.strftime(time_fmt)],
            point=test_point,
            df_ref_geo=df_ref_geo,
        )
        res.append(doc)
        sleep(2)

    dest_dir = "data"
    makedirs(dest_dir, exist_ok=True)
    df = pd.concat(res)

    df.to_csv(path.join(dest_dir, f"df_res_{station_id}.csv"), index=False)

    ""

    # TODO : se baser sur station la plus proche AVEC données publique (ou encore d'actualité...)