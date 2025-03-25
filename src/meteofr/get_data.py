import numpy as np
import pandas as pd
import requests
import numba

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


def get_ref(dep: int, prm: str) -> pd.DataFrame:
    """To get data from API into a DataFrame result

    Args:
        dep (int): id of departement
        prm (str): parameter (e.g: temperature)

    Returns:
        pd.DataFrame: result dataframe
    """
    from json import loads
    
    request_url = (
        "https://public-api.meteofrance.fr/public/DPClim/v1/liste-stations/quotidienne?"
        f"id-departement={dep}&parametre={prm}"
    )
    
    client = Client()

    client.session.headers.update({"Accept": "application/json"})

    response = client.request(
        "GET",
        request_url,
    )

    df_ref_geo: pd.DataFrame = pd.json_normalize(loads(response.content))

    return df_ref_geo

def get_all_ref(list_dep: list[int]) -> pd.DataFrame:
    # utiliser platformdirs pour la gestion de l'arbo du cache par OS ?
    # pour éviter les redownload
    df_list = []
    # for i in range(1, 96):
    for i in list_dep:
        df_list.append(get_ref(dep = i, prm="temperature"))

    df_ref_geo = pd.concat(df_list)
    
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
    """Numba attempt to speed up things"""
    n = vec.shape[0]
    m = ref.shape[0]
    res = np.zeros(shape=(n, m))
    for i in range(n):
        lat1, lon1 = vec[i, :]
        for j in range(m):
            lat2, lon2 = ref[j, :]
            res[i, j] = hvs(lat1, lon1, lat2, lon2)

    return res

@numba.jit
def get_closest_point(
    vec: np.ndarray, ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mat = get_dist(vec, ref)

    return mat.argmin(axis=1), mat.min(axis=1)

def get_weather(point = None, id = None, name=None):
    
    assert not all([point is None, id is None, name is None]), "1 parameter point, id or name must be given"
    
    df_ref_geo = get_all_ref()

if __name__ == "__main__":

    # on veut
    # donner un site/coord/... + temporalité et récoupérer les infos requises
    
    test_point = (45.932050, 2.000847)
    
    # get_weather()