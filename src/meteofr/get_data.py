import numpy as np
import pandas as pd
import requests
import numba # type; ignore
from functools import cache
import logging
from typing import Any

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


def get_rqt(request_url: str, format_result: str = "json") -> Any:
    from json import loads

    client = Client()
    client.session.headers.update({"Accept": "application/json"})
    response = client.request(
        "GET",
        request_url,
    )

    if response.status_code >= 200 and response.status_code < 300:
        pass
    else:
        raise ValueError(
            f"response.status_code: {response.status_code}\n{response.url}\n{response.text}"
        )

    if format_result == "pd":
        df: pd.DataFrame = pd.json_normalize(loads(response.content))
        return df
    elif format_result == "json":
        return loads(response.content)
    else:
        return response


@cache
def get_ref(dep: str, prm: str) -> pd.DataFrame:
    """To get data from API into a DataFrame result

    Args:
        dep (int): id of departement
        prm (str): parameter (e.g: temperature)

    Returns:
        pd.DataFrame: result dataframe
    """

    logger.debug(f"begin get ref: {dep}, {prm}")
    request_url = (
        f"{url_api}/liste-stations/quotidienne?id-departement={dep}&parametre={prm}"
    )

    return get_rqt(request_url=request_url, pd_result=True)


def get_all_ref(list_dep: list[str] = list_dep) -> pd.DataFrame:
    from pathlib import Path
    from os import makedirs, path
    from tqdm import tqdm

    logger.debug("begin get all ref")
    dir_cache = Path.home().joinpath(".meteofr")

    makedirs(dir_cache, exist_ok=True)

    df_list = []
    for i in tqdm(list_dep):
        df_list.append(get_ref(dep=i, prm="temperature"))

    df_ref_geo = pd.concat(df_list)
    df_ref_geo.to_csv(path.join(dir_cache, "ref.csv"))

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


def get_weather(point=None, id=None, name=None):
    assert not all([point is None, id is None, name is None]), (
        "1 parameter point, id or name must be given"
    )

    # get all stations
    df_ref_geo = get_all_ref(list_dep=list_dep[:10])

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
        #     ref=df_ref_geo[["lat", "lon"]].to_numpy(dtype=(np.float64, np.float64)),)

        # d, idx = mat.argmin(axis=1), mat.min(axis=1)

    #     curl -X 'GET' \
    #   'https://public-api.meteofrance.fr/public/DPClim/v1/commande-station/quotidienne?id-station=02035001&date-deb-periode=2025-01-01T00%3A00%3A00Z&date-fin-periode=2025-03-27T00%3A00%3A00Z' \
    #   -H 'accept: */*'

    # v1/commande-station/quotidienne?id-station=02035001&date-deb-periode=2025-01-01T00%3A00%3A00Z&date-fin-periode=2025-03-27T00%3A00%3A00Z

    # get weather data from closest station
    # station_id = df_ref_geo.iloc[idx][["id"]]
    # api = "v1/commande-station/quotidienne"
    # url_point_weather = f"{url_api}/{api}?"
    url_point_weather = """https://public-api.meteofrance.fr/public/DPClim/v1/commande-station/quotidienne?id-station=02035001&date-deb-periode=2025-01-01T00%3A00%3A00Z&date-fin-periode=2025-01-10T00%3A00%3A00Z"""

    id_rqt = get_rqt(url_point_weather)
    # {
    # "elaboreProduitAvecDemandeResponse": {
    #     "return": "2025001387957"
    # }
    # }

    # api call
    # https://public-api.meteofrance.fr/public/DPClim/v1/commande/fichier?id-cmde=2025001387957

    # code 500 : commande échouée... voilou bonne journée

    # results
    id_cmde = id_rqt["elaboreProduitAvecDemandeResponse"]["return"]
    # url_rqt = "https://public-api.meteofrance.fr/public/DPClim/v1/commande/fichier?id-cmde=2025001387957"
    url_rqt = f"https://public-api.meteofrance.fr/public/DPClim/v1/commande/fichier?id-cmde={id_cmde}"
    # df = get_rqt(url_rqt, pd_result=True)
    doc = get_rqt(url_rqt, pd_result=False)

    # df

    # request_url = url_rqt

    return


if __name__ == "__main__":
    # on veut
    # donner un site/coord/... + temporalité et récupérer les infos requises

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    test_point = (45.932050, 2.000847)

    logger.info("Howdy !")

    get_weather(point=test_point)
